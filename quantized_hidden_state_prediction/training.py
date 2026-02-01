import sys
import time

from functools import partial
from pathlib import PosixPath
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import os

import torch

from omegaconf import DictConfig, ListConfig, OmegaConf
from evaluation.natural_language_levels_evaluation import nl_learning_levels_evaluation
from evaluation.pfa_evaluation import pfa_training_evaluation

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate_packed, padded_collate_sft
from dataset_classes.packing_on_the_fly import PackedOnTheFlyDataset
from torchtune.datasets import ConcatDataset
from dataset_classes.stochastic_languages import LearningLevelsPFADataset
from torchtune.datasets._text_completion import TextCompletionDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
from utils.meters import MultiMeter
from utils.misc import seed_everything
from huggingface_hub import snapshot_download

from tqdm import tqdm
import wandb

log = utils.get_logger("DEBUG")


class SelfPredictionTrainingRecipeDistributed(FTRecipeInterface):
    """
    Full training recipe for language models using the Self-Prediction (PHi) methodology.

    This recipe is designed to train and evaluate Transformer or LSTM models
    integrated with a PHi (Prediction of Hidden states) layer. It supports
    distributed training (FSDP), training from scratch, or fine-tuning only the
    PHi-specific components on a pre-trained model.

    Core Method:
    The recipe's primary function is to train models not just on next-token prediction,
    but also on an auxiliary objective: predicting their own future hidden states.
    This is achieved by incorporating a `PHiLayer` into the model, which generates
    additional losses (e.g., PHi loss, self-critic loss). These are collected
    and added to the main training objective.

    Key Features:
    - **Self-Prediction (PHi) Training**:
        - **Flexible Training Modes**: Supports training the entire model, or freezing the base
          model and training only the PHi layer parameters (post-hoc training). This is
          controlled by the `train_whole_model` flag.
        - **Custom Loss Objective**: Combines the standard next-token prediction loss with
          losses from the PHi layer. The next-token loss can be disabled via the
          `ignore_main_training_loss` flag to train purely on the self-prediction signal.

    - **Specialized Evaluation**:
        - Includes periodic evaluation loops that call specific functions like
          `pfa_training_evaluation` and `nl_learning_levels_evaluation`.
        - These evaluations compute custom metrics tailored to the research, such as the
          "interestingness ratio" (comparing PHi loss on complex vs. simple tasks) and
          "length generalization" for procedural languages.

    - **Custom Data Handling**:
        - Natively supports custom procedural datasets like `LearningLevelsPFADataset`.
        - Implements an on-the-fly packing strategy (`PackedOnTheFlyDataset`) for
          efficiently handling sequences of varying lengths without pre-processing.

    - **Standard LLM Training Features**:
        - **FSDP**: Distributed training is supported via PyTorch's FSDP APIs.
        - **Precision**: Supports `fp32` and `bf16` precision modes.
        - **Activation Checkpointing**: Can be enabled to reduce memory usage.
        - **Gradient Accumulation**: Simulates larger batch sizes.
        - **Checkpointing**: Saves model and recipe state, with options to resume.
        - **Logging**: Supports multiple loggers, including WandB and TensorBoard.

    Args:
        cfg (DictConfig): An OmegaConf object parsed from a YAML file, containing the
            full configuration for the model, dataset, optimizer, and training run.

    Raises:
        ValueError: If `dtype` is set to fp16, or if both `train_from_scratch`
            and `resume_from_checkpoint` are enabled.
        RuntimeError: If using a fused optimizer on CPU with an unsupported PyTorch version.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError(
                "Using fused optimizer on CPU is only supported in PyTorch nightly."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._checkpoint_every_n_steps = cfg.get("checkpoint_every_n_steps", 1000)
        self._evaluate_n_datapoints = cfg.get("evaluate_n_datapoints", 1000)    
        self._evaluate_every_n_steps = cfg.get("evaluate_every_n_steps", 1000)
        self._save_optimizer_state = cfg.get("save_optimizer_state", True)
        self._overwrite_checkpoints = cfg.get("overwrite_checkpoints", True)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        self._world_size, rank = utils.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._train_from_scratch = cfg.get("train_from_scratch", False)
        if self._train_from_scratch:
            cfg.checkpointer.checkpoint_dir = cfg.checkpointer.output_dir
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        if self._resume_from_checkpoint and self._train_from_scratch:
            raise ValueError(
                "Both train_from_scratch and resume_from_checkpoint cannot be set to True."
            )
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        seed_everything(self.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_total_steps = cfg.max_total_steps
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._train_whole_model = cfg.get("train_whole_model", True)
        self._ignore_main_training_loss = cfg.get("ignore_main_training_loss", False)
        self._ic_generalization_eval = cfg.get("ic_generalization_eval", False)
        self.info_bottleneck = cfg.get("model", {}).get("self_prediction_information_bottleneck", "continuous")
        self.cfg = cfg

        # assertions for configurations
        if self.info_bottleneck == 'continuous':
            assert cfg.get("model", {}).get("self_prediction_module", None) is None, ( 
                "Config Error: Self prediction module must be None when using continuous information bottleneck."
            )
        elif self.info_bottleneck == 'residual_quantize' or self.info_bottleneck == 'vector_quantize':
            assert cfg.get("model", {}).get("self_prediction_module", None) is not None, (
                "Config Error: Self prediction module must be defined when using quantized information bottleneck."
            )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        #

        if self._train_from_scratch:
            cfg_checkpointer["checkpoint_dir"] = None
            cfg_checkpointer["checkpoint_files"] = None
        else:
            if not os.path.exists(cfg_checkpointer["checkpoint_dir"]):
                print("Model not found locally. Downloading from Hugging Face...")
                snapshot_download(
                    repo_id=cfg_checkpointer["checkpoint_dir"].replace("checkpoints/", "meta-llama/"),
                    local_dir=cfg_checkpointer["checkpoint_dir"],
                    local_dir_use_symlinks=False,
                )

        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        if not self._train_from_scratch:
            checkpoint_dict = self._checkpointer.load_checkpoint()
        else:
            checkpoint_dict = None

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, sampler, and dataloader.
        """

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        ckpt_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        if self._is_rank_zero:
            output_dir = self._checkpointer._output_dir
            if "$WANDB_RUN_ID" in output_dir.name:
                try:
                    run_id = self._metric_logger._wandb.run.id
                    output_dir = str(output_dir).replace("$WANDB_RUN_ID", run_id)
                    output_dir = PosixPath(output_dir)

                    self.pickle_dir = output_dir

                    print("run id: ", run_id)
                except AttributeError:
                    print(
                        "When using the '$WANDB_RUN_ID' variable in the output folder name, the logger should be wandb"
                    )
            self._checkpointer._output_dir = output_dir

        self._compile = cfg.get("compile", False)
        if self._world_size == 1:
            self._model = self._setup_model_single_device(
                cfg_model=cfg.model,
                enable_activation_checkpointing=cfg.enable_activation_checkpointing,
                compile_model=self._compile,
                model_state_dict=ckpt_dict[training.MODEL_KEY]
                if ckpt_dict is not None
                else None,
            )
        else:
            self._model = self._setup_model_distributed(
                cfg_model=cfg.model,
                enable_activation_checkpointing=cfg.enable_activation_checkpointing,
                custom_sharded_layers=cfg.get("custom_sharded_layers", None),
                fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
                reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
                model_state_dict=ckpt_dict[training.MODEL_KEY]
                if ckpt_dict is not None
                else None,
                ac_mode=cfg.get("ac_mode", None),
                ac_option=cfg.get("ac_option", None),
            )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        if hasattr(self._model, "pad_token_id"):
            self._model.pad_token_id = self._tokenizer.pad_id

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=ckpt_dict[training.OPT_KEY]
            if self._resume_from_checkpoint
            else None,
        )

        log.info(f'information bottleneck: {self.info_bottleneck}')
        log.info(f'phi loss factor: {cfg.model.phi_loss_factor}')
        log.info(f'self critic loss factor: {cfg.model.self_critic_loss_factor}')
        if self.info_bottleneck == 'residual_quantize' or \
                self.info_bottleneck == 'vector_quantize' or \
                self.info_bottleneck == 'residual_quantize_qinco' or \
                self.info_bottleneck == 'gumbel_quantize':
            log.info(f'reconstruction loss factor: {cfg.model.self_prediction_module.reconstruction_loss_factor}')
            self.get_temperature = config.instantiate(cfg.temperature_scheduler)
            self.reconstruction_loss_factor = cfg.model.self_prediction_module.reconstruction_loss_factor
            if self.info_bottleneck == 'vector_quantize':
                self.get_beta = config.instantiate(cfg.beta_scheduler)
        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        # if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
        if 'Chunked' in self._loss_fn.__class__.__name__:
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        if self._is_rank_zero:
            log.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
        )

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        if self._sampler is None:
            self._steps_per_epoch = self._estimate_steps_per_epoch()
        else:
            self._steps_per_epoch = (
                len(self._dataloader) // self._gradient_accumulation_steps
            )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        lr_scheduler_cfg = cfg.get("lr_scheduler", None)
        if lr_scheduler_cfg is None:
            lr_scheduler_cfg = DictConfig(
                {
                    "_component_": "torchtune.modules.get_constant_schedule_with_warmpup",
                    "num_warmup_steps": 0,
                }
            )

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=lr_scheduler_cfg,
            num_training_steps=self.max_total_steps,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        if self._is_rank_zero:
            log.info(f" Profiler config after instantiation: {profiler_cfg}")

            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model_single_device(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling activation checkpointing.
        """
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        if model_state_dict is not None:
            if training.MODEL_KEY in model_state_dict:
                model_state_dict = model_state_dict[training.MODEL_KEY]
            if hasattr(self._checkpointer, "_self_prediction_checkpoint_path"):
                self_prediction_path_exists = (
                    self._checkpointer._self_prediction_checkpoint_path is not None
                )
            else:
                self_prediction_path_exists = False
            model.load_state_dict(model_state_dict, strict=self_prediction_path_exists)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        log.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type == "cuda":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return model

    def _setup_model_distributed(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        custom_sharded_layers: Optional[List[str]],
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        if self._is_rank_zero:
            log.info(
                "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ..."
            )
            init_start = time.perf_counter()

        if model_state_dict is None or not self._train_whole_model:
            with training.set_default_dtype(self._dtype):
                model = config.instantiate(cfg_model)
        else:
            with training.set_default_dtype(self._dtype), torch.device("meta"):
                model = config.instantiate(cfg_model)

        if model_state_dict is None:
            model_state_dict = model.state_dict()
        elif not self._train_whole_model:
            random_init_state_dict = model.state_dict()
            for key in random_init_state_dict:
                if key not in model_state_dict:
                    print(f"use random init for {key}: {random_init_state_dict[key]}")
                    model_state_dict[key] = random_init_state_dict[key]

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding, we can condition on either the module or its name
        # Shard conditions should be callables taking name (relative to model root)
        # and the module itself and returning a bool on whether to shard the given module
        fsdp_shard_conditions = []

        # Shard transformer decoder layers (or AC-wrapped versions)
        # Alternatively we could condition on the module type (TransformerDecoder or CheckpointWrapper)
        # But directly using the name is more concise
        def _is_layer_fqn(s: str) -> bool:
            """
            Return True for layers.i and False for all other module names
            Covers sharding for both AC-wrapped and non-AC-wrapped modules in one shot
            """
            s_list = s.split(".")
            return len(s_list) == 2 and s_list[0] == "layers" and str.isdigit(s_list[1])

        fsdp_shard_conditions = [lambda n, m: _is_layer_fqn(n)]

        # If wrapping any layers separately, we can add another shard condition
        # A layer will be sharded if any of the fsdp_shard_conditions are met
        if custom_sharded_layers:
            fsdp_shard_conditions += [lambda n, m: n in custom_sharded_layers]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        self_prediction_path_exists = (
            self._checkpointer._self_prediction_checkpoint_path is not None
        )
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            self._is_rank_zero,
            strict=self_prediction_path_exists,
            cpu_offload=fsdp_cpu_offload,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero:
            num_params = sum(p.numel() for p in model.parameters())
            log.info(f"Number of parameters: {num_params}")

            log.info(
                f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs"
            )
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        if self._train_whole_model:
            params = self._model.parameters()
        else:
            params = self._model.self_prediction_layer.parameters()
            # all parameters that are not from the self_prediction_layer do not require gradients
            for name, param in self._model.named_parameters():
                if "self_prediction_layer" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        optimizer = config.instantiate(cfg_optimizer, params)
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                optimizer,
                opt_state_dict,
                self._device,
            )

        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All dataset_classes related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable dataset_classes and streaming dataset_classes are not supported.
        """
        world_size, rank = utils.get_world_size_and_rank()

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = False
        else:
            packed = cfg_dataset.get("packed", False)

            packed_on_the_fly = cfg_dataset.pop("packed_on_the_fly", False)
            packed_sequence_length = cfg_dataset.pop("packed_sequence_length", 2048)
            split_across_pack = cfg_dataset.pop("split_across_pack", False)
            num_workers = cfg_dataset.pop("num_workers", 8)

            if packed_on_the_fly and "packed" in cfg_dataset:
                cfg_dataset["packed"] = False
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            if packed_on_the_fly:
                ds = PackedOnTheFlyDataset(
                    ds,
                    max_seq_len=packed_sequence_length,
                    padding_idx=self._tokenizer.pad_id,
                    world_size=world_size,
                    rank=rank,
                    permute_indices=shuffle,
                    split_across_pack=split_across_pack,
                )
                packed = True

        if packed_on_the_fly:
            dataloader = DataLoader(
                dataset=ds,
                batch_size=batch_size,
                num_workers=num_workers,
                worker_init_fn=ds._worker_init_fn,
                collate_fn=partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else partial(
                    padded_collate_packed,
                ),
            )
            log.info("On the fly packing & tokenization Dataset is initialized.")
            return None, dataloader
        else:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0
            )
            dataloader = DataLoader(
                dataset=ds,
                batch_size=batch_size,
                sampler=sampler,
                # dropping last avoids shape issues with compile + flex attention
                drop_last=cfg_dataset.get("drop_last", True),
                collate_fn=partial(
                    padded_collate_sft,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else partial(
                    padded_collate_packed,
                ),
            )

            if self._is_rank_zero:
                log.info("Dataset and Sampler are initialized.")

            return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
        intermediate_checkpoint: bool = True,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the model weights and recipe state in
        different checkpoint files. To correctly resume training from an intermediate checkpoint,
        the model weights and recipe state must be provided.
        """
        # final dict passed onto the checkpointer
        if self._checkpointer is None:
            if self._is_rank_zero:
                log.info("Checkpointer is not initialized. Skipping checkpointing.")
            return
        checkpoint_dict = {}

        if self._world_size == 1:
            if self._train_whole_model:
                cpu_state_dict = {
                    training.MODEL_KEY: {
                        k: v.cpu() for k, v in self._model.state_dict().items()
                    }
                }
            else:
                cpu_state_dict = {
                    training.MODEL_KEY: {
                        k: v.cpu()
                        for k, v in self._model.self_prediction_layer.state_dict().items()
                    }
                }
        else:
            # To prevent GPU memory from spiking during checkpoint save,
            # we consolidate the full model and optim state dicts on CPU for rank 0
            if self._train_whole_model:
                cpu_state_dict = training.get_full_model_state_dict(
                    self._model,
                    self._is_rank_zero,
                    device=self._device,
                )
            else:
                cpu_state_dict = training.get_full_model_state_dict(
                    self._model.self_prediction_layer,
                    self._is_rank_zero,
                    device=self._device,
                )

        if intermediate_checkpoint:
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            checkpoint_dict.update(cpu_state_dict)

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )

            # save also the cfg as a yaml file using OmegaConf
            yaml_path = f"{self._checkpointer._output_dir}/config.yaml"
            with open(yaml_path, "w") as f:
                OmegaConf.save(self.cfg, f.name)

    def _estimate_steps_per_epoch(self, num_batches=5) -> int:
        dataset_length = len(self._dataloader.dataset.ds)
        num_examples = 0
        for i, batch in enumerate(self._dataloader):
            if i >= num_batches:
                break
            # count the number of times the position sequences go back to 0 (and it's not padding)
            num_zeros = torch.sum(
                torch.logical_and(
                    batch["input_pos"] == 0, batch["tokens"] != self._tokenizer.pad_id
                )
            ).item()

            num_examples += num_zeros
        examples_per_batch = num_examples / num_batches
        steps_per_epoch = int(
            dataset_length / (examples_per_batch * self._gradient_accumulation_steps)
        )
        return steps_per_epoch

    def _loss_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Shape [b, s], needed for the loss not the model
        labels = batch.pop("labels")

        logits = self._model(**batch)

        # Shift labels to compute loss
        # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
        # But this way we don't need to slice the logits. We just add an ignore index to labels.
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))
        
        # Compute loss
        loss = self._loss_fn(logits, labels)
        assert loss.numel() == 1, f"Loss should be a scalar. Use a loss function that aggregates the loss!"
        # free logits otherwise it peaks backward memory
        del logits

        additional_logging_losses = {}
        if callable(getattr(self._model, "get_additional_losses", None)):
            next_token_prediction_loss = loss.clone()
            if self._ignore_main_training_loss:
                loss = 0.0

            additional_training_losses, additional_logging_losses = self._model.get_additional_losses()
            for loss_name, loss_val in additional_training_losses.items():
                loss += loss_val
            additional_logging_losses[
                "next token prediction losses"
            ] = next_token_prediction_loss.detach()

        return loss, additional_logging_losses

    def train(self, save_at_the_end=True) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = utils.get_world_size_and_rank()

        # zero out the gradients before starting training
        # self._optimizer.zero_grad()
        self._model.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0
        cumulative_tokens = 0
        meter = MultiMeter()

        # for debugging, ignore, delete later
        try:
            self._model.self_prediction_layer.log_hist = False # Delete later, temp, debug
        except:
            pass

        self._profiler.start()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            # Update the sampler to ensure dataset_classes is correctly shuffled across epochs
            # in case shuffle is True
            if self._sampler is not None:
                self._sampler.set_epoch(curr_epoch)

            pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
            for idx, batch in enumerate(self._dataloader):
                if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                ):
                    torch.cuda.memory._record_memory_history()

                # modified loss step
                utils.batch_to_device(batch, self._device)
                new_tokens = batch["tokens"].numel()
                num_tokens += new_tokens
                cumulative_tokens += (
                    new_tokens * world_size
                )  # this might be an overestimate since all padding tokens are counted

                # only assign temperature when using gumbel quantization
                if self.info_bottleneck == 'residual_quantize' \
                        or self.info_bottleneck == 'vector_quantize' \
                        or self.info_bottleneck == 'residual_quantize_qinco' \
                        or self.info_bottleneck == 'gumbel_quantize':
                    self._model.self_prediction_layer.quantizer.temperature = self.get_temperature(self.global_step)
                    self._model.self_prediction_layer.reconstruction_loss_factor = self.reconstruction_loss_factor
                    if self.info_bottleneck == 'vector_quantize':
                        self._model.self_prediction_layer.quantizer.kld_scale = self.get_beta(self.global_step)

                    # for debugging, ignore, delete later
                    try:
                        if idx % 1000 == 0:
                            self._model.self_prediction_layer.log_hist = True
                    except:
                        pass
                loss, sub_losses_dict = self._loss_step(batch)

                sub_losses_dict = {k: v.item() for k, v in sub_losses_dict.items()}

                # join the loss with the sub_losses_dict
                meter.update({"loss": loss.item()} | sub_losses_dict)

                loss = loss / self._gradient_accumulation_steps
                running_loss += loss.item()
                loss.backward()

                # Step with optimizer
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )

                    self._optimizer.step()
                    # self._optimizer.zero_grad(set_to_none=True)
                    self._model.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # debugging
                    # codebook0_after = self._model.self_prediction_layer.quantizer.codebooks[0].weight.detach().clone()
                    # codebook1_after = self._model.self_prediction_layer.quantizer.codebooks[1].weight.detach().clone()
                    # print("Max change:", (codebook0_after - codebook0_before).abs().max())
                    # print("Max change:", (codebook1_after - codebook1_before).abs().max())

                    loss_to_log = running_loss
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {loss_to_log}"
                    )

                    if self.cfg.get("model", {}).get("self_prediction_module", None) is not None and self.cfg.model.self_prediction_information_bottleneck == 'residual_quantize' and \
                        self.cfg.model.self_prediction_module.save_codebooks_for_qinco and \
                        self.global_step % self.cfg.model.self_prediction_module.save_every_n_steps == 0:
                        
                        self._model.self_prediction_layer.quantizer.save_codebooks(self.global_step)


                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "loss": loss_to_log,
                            "lr": self._optimizer.param_groups[0]["lr"],
                            "tokens_per_second_per_gpu": num_tokens / time_per_step,
                            "tokens": cumulative_tokens,
                            "epoch": curr_epoch,
                        }
                        for key in meter.meters.keys():
                            log_dict[key] = meter.meters[key].avg
                        meter.reset()

                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    if self.global_step % self._evaluate_every_n_steps == 0:
                        print("Evaluating")
                        if 'learning_levels_pfa' in self.cfg.dataset._component_:
                            plotly_figure_dict, eval_values_dict = pfa_training_evaluation(
                                self,
                                num_datapoints=self._evaluate_n_datapoints,
                                ic_generalization_evaluation=self._ic_generalization_eval
                            )
                        else:
                            plotly_figure_dict, eval_values_dict = nl_learning_levels_evaluation(self, self._evaluate_n_datapoints)

                        log_dict = plotly_figure_dict
                        for key, value in eval_values_dict.items():
                            log_dict[key] = value

                        if self._is_rank_zero:
                            self._metric_logger.log_dict(
                                log_dict,
                                step=self.global_step,
                            )

                    if self.global_step % self._checkpoint_every_n_steps == 0:
                        self.save_checkpoint(
                            epoch=self.global_step
                            if not self._overwrite_checkpoints
                            else None,
                            intermediate_checkpoint=self._save_optimizer_state,
                        )

                    # Reset running stats for the next step
                    running_loss = 0
                    num_tokens = 0
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                    if self.global_step >= self.max_total_steps:
                        break

            if self.epochs_run == 0:
                self._steps_per_epoch = self.global_step

            self.epochs_run += 1

            if self.global_step >= self.max_total_steps:
                break

        self._profiler.stop()

        if save_at_the_end:
            self.save_checkpoint(
                epoch=self.global_step if not self._overwrite_checkpoints else None,
                intermediate_checkpoint=False,
            )

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        if self._world_size > 1:
            destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    print("start recipe_main")
    if training.is_distributed():
        init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
        if cfg.get("fsdp_cpu_offload", False):
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()
    else:
        cfg.nproc_per_node = 1

    config.log_config(recipe_name="FullTrainingRecipeDistributed", cfg=cfg)

    cfg.evaluate_every_n_steps = 10

    wandb.init(
        project="self-prediction",
    )
    sweep_cfg = wandb.config
    if "seed" in sweep_cfg:
        cfg.seed = sweep_cfg.seed

    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()

    wandb.finish()


if __name__ == "__main__":
    sys.exit(recipe_main())
