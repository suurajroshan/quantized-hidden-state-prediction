# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import json
import os

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from enum import Enum

import torch
from safetensors.torch import save_file
from torchtune import training

from torchtune.models import convert_weights
from torchtune.models.phi3._convert_weights import phi3_hf_to_tune, phi3_tune_to_hf
from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune, qwen2_tune_to_hf
from torchtune.rlhf.utils import reward_hf_to_tune, reward_tune_to_hf
from torchtune.training.checkpointing._utils import (
    FormattedCheckpointFiles,
    get_path,
    # ModelType,
    safe_torch_load,
    # save_config
)
from torchtune.training.checkpointing._checkpointer import _CheckpointerInterface

from torchtune.utils._logging import get_logger, log_rank_zero

logger = get_logger("DEBUG")

class ModelType(Enum):
    """ModelType is used by the checkpointer to distinguish between different model architectures.

    If you are adding a new model that follows a different format than those in the repo already,
    you can add a new ModelType to gate on weight conversion logic unique to that model.

    Attributes:
        GEMMA (str): Gemma family of models. See :func:`~torchtune.models.gemma.gemma`
        GEMMA2 (str): Gemma 2 family of models. See :func:`~torchtune.models.gemma2.gemma2`
        LLAMA2 (str): Llama2 family of models. See :func:`~torchtune.models.llama2.llama2`
        LLAMA3 (str): Llama3 family of models. See :func:`~torchtune.models.llama3.llama3`
        LLAMA3_2 (str): Llama3.2 family of models. See :func:`~torchtune.models.llama3_2.llama3_2`
        LLAMA3_VISION (str): LLama3 vision family of models. See :func:`~torchtune.models.llama3_2_vision.llama3_2_vision_decoder`
        MISTRAL (str): Mistral family of models. See :func:`~torchtune.models.mistral.mistral`
        PHI3_MINI (str): Phi-3 family of models. See :func:`~torchtune.models.phi3.phi3`
        PHI4 (str): Phi-4 family of models. See :func:`~torchtune.models.phi4.phi4`
        REWARD (str): A Llama2, Llama3, or Mistral model with a classification head projecting
            to a single class for reward modelling.
            See :func:`~torchtune.models.mistral.mistral_reward_7b` or :func:`~torchtune.models.llama2.llama2_reward_7b`
        QWEN2 (str): Qwen2 family of models. See :func:`~torchtune.models.qwen2.qwen2`
        CLIP_TEXT (str): CLIP text encoder. See :func:`~torchtune.models.clip.clip_text_encoder_large`
        T5_ENCODER (str): T5 text encoder. See :func:`~torchtune.models.t5.t5_v1_1_xxl_encoder`

    Example:
        >>> # Usage in a checkpointer class
        >>> def load_checkpoint(self, ...):
        >>>     ...
        >>>     if self._model_type == MY_NEW_MODEL:
        >>>         state_dict = my_custom_state_dict_mapping(state_dict)
    """

    GEMMA: str = "gemma"
    GEMMA2: str = "gemma2"
    LLAMA2: str = "llama2"
    LLAMA3: str = "llama3"
    LLAMA3_2: str = "llama3_2"
    LLAMA3_SELF_PREDICTION: str = "llama3_self_prediction"
    LLAMA3_SELF_PREDICTION_QUANTIZED: str = "llama3_self_prediction_quantized"
    LLAMA3_VISION: str = "llama3_vision"
    MISTRAL: str = "mistral"
    PHI3_MINI: str = "phi3_mini"
    PHI4: str = "phi4"
    REWARD: str = "reward"
    QWEN2: str = "qwen2"
    CLIP_TEXT: str = "clip_text"
    T5_ENCODER: str = "t5_encoder"


class FullModelTorchTuneCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in a format compatible with
    torchtune. No conversion of weights is required.

    Currently this supports reading a single checkpoint file only. This will likely change as
    we add support for larger models.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False

    Raises:
        ValueError: If more than one checkpoint file is provided
        ValueError: If the checkpoint file does not have a .pt extension
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None


    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        if checkpoint_dir is not None:
            if len(checkpoint_files) != 1:
                raise ValueError(
                    "Currently we only support reading from a single torchtune checkpoint file. "
                    f"Got {len(checkpoint_files)} files instead."
                )

            self._checkpoint_dir = Path(checkpoint_dir)
            self._checkpoint_path = get_path(self._checkpoint_dir, checkpoint_files[0])

            if not self._checkpoint_path.suffix == ".pt":
                raise ValueError(
                    f"Checkpoint file {self._checkpoint_path} is not a valid checkpoint file. "
                    "Checkpointer expects a valid .pt file."
                )
        else:
            self._checkpoint_dir = None
            self._checkpoint_path = None

        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        self._resume_from_checkpoint = resume_from_checkpoint
        self._model_type = model_type
        self._output_dir = Path(output_dir)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def load_checkpoint(self, weights_only: bool = True) -> Dict[str, Any]:
        """
        Load torchtune checkpoint from file. Currently only loading from a single file is supported.

        The output state_dict has the following format, with keys other than "model" only present if
        ``resume_from_checkpoint`` is True:

        >>>     {
        >>>         "model": {
        >>>             "key_1": weight
        >>>             ...
        >>>         },
        >>>         "optimizer": {...},
        >>>         ...
        >>>     }

        Args:
            weights_only (bool): flag passed down to torch.load. We expose this, because quantized models
                cannot be loaded with weights_only=True

        Returns:
            Dict[str, Any]: state_dict from the input checkpoint
        """
        state_dict: Dict[str:Any] = {}
        state_dict[training.MODEL_KEY] = safe_torch_load(
            self._checkpoint_path, weights_only=weights_only
        )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: Optional[int] = None,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save torchtune checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state. The output state dicts have the following formats:

        >>> # Model
        >>> {
        >>>     "key_1": weight
        >>>     ...
        >>> }
        >>>
        >>> # Recipe state
        >>> {
        >>>     "optimizer": ...,
        >>>     "epoch": ...,
        >>>     ...
        >>> }

        Args:
            state_dict (Dict[str, Any]): State dict with model and (optionally) recipe state
            epoch/step (int): Current epoch number or step number. This is added to the checkpoint file name to ensure
                we're not overwriting intermediate checkpoint files. If epoch is None, the checkpoint file is overwritten.
            intermediate_checkpoint (bool): If True, save an additional checkpoint file with the
                recipe state
            adapter_only (bool): If True, only save the adapter weights. Default is False


        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        self._output_dir.mkdir(exist_ok=True)
        if epoch is None:
            epoch = "last"

        # Output file is always a .pt file with the epoch number in the name
        if not adapter_only:
            checkpoint_file = Path.joinpath(
                self._output_dir, f"torchtune_model_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.MODEL_KEY], checkpoint_file)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
                f"saved to {checkpoint_file}"
            )

        if training.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        # If the recipe state needs to be output, first remove the model state dict
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )

class FullModelMetaCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in Meta's format. Examples include
    the Llama-2-7b model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b)

    Currently we support reading from a single checkpoint file only. Support for reading from
    sharded checkpoints is WIP.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Currently this checkpointer only
            supports loading a single checkpoint file.
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False

    Raises:
        ValueError: If ``checkpoint_files`` is not a list of length 1
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        self_prediction_checkpoint_dir: Optional[str] = None,
        self_prediction_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        if checkpoint_dir is not None:
            if len(checkpoint_files) != 1:
                raise ValueError(
                    "Currently we only support reading from a single torchtune checkpoint file. "
                    f"Got {len(checkpoint_files)} files instead."
                )

            self._checkpoint_dir = Path(checkpoint_dir)
            self._checkpoint_path = get_path(self._checkpoint_dir, checkpoint_files[0])
        else:
            self._checkpoint_dir = None
            self._checkpoint_path = None

        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        if self_prediction_checkpoint_dir is not None:
            self._self_prediction_checkpoint_dir = Path(self_prediction_checkpoint_dir)
            self._self_prediction_checkpoint_path = get_path(
                self._self_prediction_checkpoint_dir, self_prediction_checkpoint
            )
        else:
            self._self_prediction_checkpoint_path = None

        self._resume_from_checkpoint = resume_from_checkpoint
        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
        elif self._model_type == ModelType.LLAMA3_SELF_PREDICTION:
            if "consolidated" in self._checkpoint_path.name:
                # load the original Llama model weights
                state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(
                    model_state_dict
                )
            else:
                # load torchtune model weights
                state_dict[training.MODEL_KEY] = model_state_dict
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(
                model_state_dict
            )

        if (
            self._model_type == ModelType.LLAMA3_SELF_PREDICTION
            and self._self_prediction_checkpoint_path
        ):
            self_prediction_state_dict = safe_torch_load(
                self._self_prediction_checkpoint_path
            )
            for key, value in self_prediction_state_dict.items():
                if not "self_prediction_layer" in key:
                    key = "self_prediction_layer." + key
                state_dict[training.MODEL_KEY][key] = value

        # llama3_2 has tied weights, so we need to remove the output.weight key
        if self._model_type == ModelType.LLAMA3_2 or self._model_type == ModelType.LLAMA3_SELF_PREDICTION:
            logger.info(
                "Identified model_type = Llama3_2. Ignoring output.weight in"
                " checkpoint in favor of the tok_embedding.weight"
                " tied weights."
            )
            state_dict[training.MODEL_KEY].pop("output.weight")

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save Meta checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        self._output_dir.mkdir(exist_ok=True)

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_meta,
                )

                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_meta(
                    model_state_dict
                )
            elif self._model_type == ModelType.LLAMA3_SELF_PREDICTION:
                state_dict[training.MODEL_KEY] = model_state_dict
            else:
                # llama3_2 has tied weights, so we need to add the output.weight key
                if (
                    self._model_type == ModelType.LLAMA3_2
                    and "output.weight" not in model_state_dict
                ):
                    model_state_dict["output.weight"] = model_state_dict[
                        "tok_embeddings.weight"
                    ]

                state_dict[training.MODEL_KEY] = convert_weights.tune_to_meta(
                    model_state_dict
                )

            # Output file is always a .pt file with the epoch number in the name
            checkpoint_file = Path.joinpath(
                self._output_dir, f"meta_model_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.MODEL_KEY], checkpoint_file)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
                f"saved to {checkpoint_file}"
            )

        if training.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )

class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in HF's format. For LoRA models this includes
    saving checkpoints in a format that can be loaded into PEFT via e.g. ``from_pretrained``. Examples include
    the Llama-2-7b-hf model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b-hf).

    Note:
        HF checkpoint names are usually ordered by ID (eg: 0001_of_0003, 0002_of_0003, etc.) To ensure \
        we read the files in the right order, we sort the checkpoint file names before reading.

    Note:
        Checkpoint conversion to and from HF's format requires access to model params which are \
        read directly from the ``config.json`` file. This helps ensure we either load the weights \
        correctly or error out in case of discrepancy between the HF checkpoint file and torchtune's \
        model implementations.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (Union[List[str], Dict[str, str]]): List of checkpoint files to load. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter. TODO: update this
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`

    Raises:
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: Union[List[str], Dict[str, str]],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        self_prediction_checkpoint_dir: Optional[str] = None,
        self_prediction_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        safe_serialization: bool = False,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)

        if not isinstance(checkpoint_files, List):
            formatted_checkpoint_files = FormattedCheckpointFiles.from_dict(
                checkpoint_files
            )
            checkpoint_files = formatted_checkpoint_files.build_checkpoint_filenames()
        self._checkpoint_paths = self._validate_hf_checkpoint_files(checkpoint_files)
        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        if self_prediction_checkpoint_dir is not None:
            self._self_prediction_checkpoint_dir = Path(self_prediction_checkpoint_dir)
            self._self_prediction_checkpoint_path = get_path(
                self._self_prediction_checkpoint_dir, self_prediction_checkpoint
            )
        else:
            self._self_prediction_checkpoint_path = None

        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)
        self._resume_from_checkpoint = resume_from_checkpoint
        self._safe_serialization = safe_serialization

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint
        # load
        self._weight_map: Dict[str, str] = None

        # the config.json file contains model params needed for state dict conversion
        self._config = json.loads(
            Path.joinpath(self._checkpoint_dir, "config.json").read_text()
        )

        # save config.json to output_dir
        # save_config(self._output_dir, self._config)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

    def _validate_hf_checkpoint_files(self, checkpoint_files: List[str]) -> List[Path]:
        """
        Validates that the checkpoint files exist and sorts based on ID.
        """
        checkpoint_paths: List[Path] = []
        for f in checkpoint_files:
            checkpoint_path = get_path(self._checkpoint_dir, f)
            checkpoint_paths.append(checkpoint_path)
        return sorted(checkpoint_paths)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        the appropriate convert_weights function (depending on ``self._model_type``).

        Returns:
            state_dict (Dict[str, Any]): torchtune checkpoint state dict

        Raises:
            ValueError: If the values in the input state_dict are not Tensors
        """
        self._weight_map = {}

        # merged state_dict contains keys and weights from all the checkpoint files
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # converted_state_dict is the final state_dict passed to the recipe after the
        # keys are converted into the torchtune format. This optionally also contains
        # the recipe state and adapter weights
        converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

        # _checkpoint_paths are already sorted so simply enumerate to generate the right id
        for cpt_idx, cpt_path in enumerate(self._checkpoint_paths):
            state_dict = safe_torch_load(cpt_path)
            for key, value in state_dict.items():
                # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                # will break recipe code
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Expected all values in the state dict to be torch.Tensor. "
                        f"Found {type(value)} instead."
                    )
                # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
                self._weight_map[key] = f"{cpt_idx + 1:04}"
            merged_state_dict.update(state_dict)

            # delete the state_dict to free up memory; TODO check if this del is needed
            del state_dict
            gc.collect()
        if self._model_type == ModelType.PHI3_MINI:
            log_rank_zero(
                logger=logger,
                msg="Converting Phi-3 Mini weights from HF format."
                "Note that conversion of adapter weights into PEFT format is not supported.",
            )
            converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(
                merged_state_dict
            )
        elif self._model_type == ModelType.REWARD:
            converted_state_dict[training.MODEL_KEY] = reward_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
            )
        elif self._model_type == ModelType.QWEN2:
            converted_state_dict[training.MODEL_KEY] = qwen2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
            )
        elif self._model_type == ModelType.LLAMA3_VISION:
            raise NotImplementedError(
                "Safe tensors support for LLaMA3 Vision will be added soon. Pleaee use the FullModelMetaCheckpointer for now."
            )
        else:
            converted_state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )

        if (self._model_type == ModelType.LLAMA3_SELF_PREDICTION or 
            self._model_type == ModelType.LLAMA3_SELF_PREDICTION_QUANTIZED) and self._self_prediction_checkpoint_path:
            self_prediction_state_dict = safe_torch_load(
                self._self_prediction_checkpoint_path,
            )
            for key, value in self_prediction_state_dict.items():
                if not "self_prediction_layer" in key:
                    key = "self_prediction_layer." + key
                converted_state_dict[training.MODEL_KEY][key] = value

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)
        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save HF checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state.

        The state_dict is first converted back to the HF format and then partitioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        self._output_dir.mkdir(exist_ok=True)

        # convert the state_dict back to hf format; do this inplace
        if not adapter_only:
            if self._model_type == ModelType.PHI3_MINI:
                state_dict[training.MODEL_KEY] = phi3_tune_to_hf(
                    state_dict[training.MODEL_KEY]
                )
            elif self._model_type == ModelType.REWARD:
                state_dict[training.MODEL_KEY] = reward_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                )
            elif self._model_type == ModelType.QWEN2:
                state_dict[training.MODEL_KEY] = qwen2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                raise NotImplementedError(
                    "Safe tensors support for LLaMA3 Vision will be added soon. Pleaee use the FullModelMetaCheckpointer for now."
                )
            elif self._model_type == ModelType.LLAMA3_SELF_PREDICTION:
                state_dict[training.MODEL_KEY] = state_dict[training.MODEL_KEY]
            elif self._model_type == ModelType.LLAMA3_SELF_PREDICTION_QUANTIZED:
                state_dict[training.MODEL_KEY] = state_dict[training.MODEL_KEY]
            else:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )

            # split the state_dict into separate dicts, one for each output checkpoint file
            split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
            for key, weight in state_dict[training.MODEL_KEY].items():
                if key in self._weight_map:
                    cpt_idx = self._weight_map[key]
                else:
                    cpt_idx = '0001'
                if cpt_idx not in split_state_dicts:
                    split_state_dicts[cpt_idx] = {}
                split_state_dicts[cpt_idx].update({key: weight})

            # write the partitioned state dicts to the right checkpoint file
            for cpt_idx, model_state_dict in split_state_dicts.items():
                if not self._safe_serialization:
                    output_path = Path.joinpath(
                        self._output_dir, f"hf_model_{cpt_idx}_{epoch}"
                    ).with_suffix(".pt")
                    torch.save(model_state_dict, output_path)
                else:
                    output_path = Path.joinpath(
                        self._output_dir,
                        f"model-0{cpt_idx}-of-0{list(split_state_dicts.keys())[-1]}_{epoch}",
                    ).with_suffix(".safetensors")
                    save_file(model_state_dict, output_path, metadata={"format": "pt"})
                logger.info(
                    "Model checkpoint of size "
                    f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                    f"saved to {output_path}"
                )

        if training.ADAPTER_KEY in state_dict:
            # Save torchtune format adapter weights even if we save PEFT format
            # This way we can resume no matter what (and memory footprint of adapter weights is small)
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )

            if self._model_type == ModelType.PHI3_MINI:
                logger.warning(
                    "Saving Phi-3 Mini adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            else:
                state_dict[
                    training.ADAPTER_KEY
                ] = convert_weights.tune_to_peft_adapter_weights(
                    state_dict[training.ADAPTER_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
                peft_output_path = Path.joinpath(
                    self._output_dir, "adapter_model"
                ).with_suffix(".bin")
                torch.save(state_dict[training.ADAPTER_KEY], peft_output_path)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                    f"saved to {peft_output_path}"
                )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        if training.ADAPTER_CONFIG in state_dict:
            if self._model_type == ModelType.PHI3_MINI:
                logger.warning(
                    "PEFT integration for Phi-3 Mini is not supported, skipping adapter config save"
                )
            else:
                state_dict[
                    training.ADAPTER_CONFIG
                ] = convert_weights.tune_to_peft_adapter_config(
                    state_dict[training.ADAPTER_CONFIG]
                )
                output_path = Path.joinpath(self._output_dir, "adapter_config.json")
                with open(output_path, "w") as f:
                    json.dump(state_dict[training.ADAPTER_CONFIG], f)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                    f"saved to {output_path}"
                )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY, None)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )