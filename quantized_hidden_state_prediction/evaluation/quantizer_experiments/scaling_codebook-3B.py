import os
from omegaconf import OmegaConf
from functools import partial
import json

import numpy as np
import torch
from torchtune import config
from torchtune.data import padded_collate_packed
from torch.utils.data import DataLoader
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import wandb


from dataset_classes import learning_levels_pfa_dataset, PackedOnTheFlyDataset
from evaluation.pfa_evaluation import generate_per_token_losses, decode_token_by_token, process_data, compute_losses_per_level, compute_losses_per_level_statistics
from training import SelfPredictionTrainingRecipeDistributed

def eval_models(run, dataset, num_datapoints):
    checkpoint_base = "/home/woody/iwi5/iwi5368h/models/llama_3B_PHi/learning_levels_sweep_"  # llama-0.1B
    checkpoint_path = checkpoint_base + run.id

    config_file = checkpoint_path + "/config.yaml"
    cfg = OmegaConf.load(config_file)
    if cfg.train_whole_model:
        print("load full model")
        cfg.checkpointer.checkpoint_dir = checkpoint_path
        cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    else:
        cfg.checkpointer._self_prediction_checkpoint_dir = checkpoint_path
        cfg.checkpointer._self_prediction_checkpoint = "meta_model_None.pt"
        
    cfg.checkpointer.output_dir = "/home/woody/iwi5/iwi5368h/models/scaling_eval"
    cfg.metric_logger.mode = 'disabled'
    cfg.train_from_scratch = False
    cfg.compile = False
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)

    recipe._model.eval()
    datapoints = process_data(recipe,num_datapoints=num_datapoints,dataset=dataset,batch_size=cfg.batch_size,)

    losses_dict = {
        loss: [] for loss in losses
    }
    for d in datapoints:
        for loss in losses:
                losses_len = len(d[loss])
                losses_dict[loss].append(d[loss])

    for loss in losses:
        c = np.concatenate(losses_dict[loss])
        losses_dict[loss] = c

    loss_stats = {
        loss: {} for loss in losses
    }

    for loss in losses:
        c = losses_dict[loss]
        loss_stats[loss]["mean"] = float(c.mean())
        loss_stats[loss]["median"] = float(np.median(c))
        loss_stats[loss]["std"] = float(c.std())
        loss_stats[loss]["num"] = len(c)
        loss_stats[loss]["std_err"] = float(c.std() / (
            len(c) ** 0.5
        ))
    return loss_stats

if __name__ == "__main__":
    losses = ['next_token_losses', 'phi_losses0', 'latent_entropy0', 'latent_losses0']
    levels = (0,1,2,3,4)
    num_datapoints = 100

    api = wandb.Api()
    runs = api.runs("hidden-state-predictions/llama-0.1-gumbel-scaling-codebook")

    random_run_dataset = np.random.randint(len(runs))

    checkpoint_base = "/home/woody/iwi5/iwi5368h/models/llama_0.1B_PHi/learning_levels_sweep_"  # llama-0.1B
    checkpoint_path = checkpoint_base + runs[random_run_dataset].id

    cfg = OmegaConf.load(checkpoint_path + "/config.yaml")
    cfg.metric_logger.mode = 'disabled'
    cfg_dataset = cfg.dataset
    cfg.token_perturbation_rate = 0.
    cfg.word_perturbation_rate = 0.

    cfg_tokenizer = cfg.tokenizer
    tokenizer = config.instantiate(cfg.tokenizer)

    cfg_dataset = cfg.dataset
    cfg_dataset.included_learning_levels = levels
    packed_on_the_fly = cfg_dataset.pop("packed_on_the_fly", False)
    packed_sequence_length = cfg_dataset.pop("packed_sequence_length", 2048)
    split_across_pack = cfg_dataset.pop("split_across_pack", False)
    num_workers = cfg_dataset.pop("num_workers", 8)

    ds = config.instantiate(cfg_dataset, tokenizer)

    results = {}
    for run in runs:
        results[run.name.split('-')[1]] = eval_models(run, ds, num_datapoints)
    with open(f"/home/woody/iwi5/iwi5368h/evaluations/pfa_evals_0.1B/scaling-run-gumbel.json", 'w') as f:
        print('saving run')
        json.dump(results, f)

