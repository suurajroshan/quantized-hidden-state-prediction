# This script evaluates multiple trained models across different random seeds 
# and over the layers

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
import pandas as pd
import wandb


from dataset_classes import learning_levels_pfa_dataset, PackedOnTheFlyDataset
from evaluation.pfa_evaluation import generate_per_token_losses, decode_token_by_token, process_data, compute_losses_per_level_statistics
from training import SelfPredictionTrainingRecipeDistributed

def compute_losses_per_level(
    datapoints,
    losses=("next_token_losses", "phi_losses"),
    prefixes = ["phi_losses", "latent_losses", "latent_entropy"],
    level_key="learning_level",
    levels=(0, 1, 2, 3, 4),
    filter_out_spaces=False,
):
    """
    Aggregates per-token losses from a list of data points, grouped by level.

    This function iterates through a list of processed data points and collects
    values for specified loss types. It uses a level key to group these loss
    values, making it easy to analyze performance across different categories or
    complexity levels.

    Args:
        datapoints (List[Dict[str, np.ndarray]]): A list of dictionaries, where
            each dictionary represents a sample and must contain token-aligned
            NumPy arrays for the specified losses and the `level_key`.
        losses (Tuple[str, ...], optional): A tuple of strings corresponding to the
            loss keys to aggregate from each data point.
            Defaults to ("next_token_losses", "phi_losses").
        level_key (str, optional): The key in each data point dictionary that
            contains the array of level indices for each token.
            Defaults to "learning_level".
        levels (Tuple[int, ...], optional): A tuple of the integer level indices
            to aggregate losses for. Defaults to (0, 1, 2, 3, 4).
        filter_out_spaces (bool, optional): If True, losses corresponding to
            space characters (ASCII 32) will be excluded from the aggregation.
            Defaults to False.

    Returns:
        Dict[str, Dict[int, np.ndarray]]: A nested dictionary where the outer keys
            are loss names and inner keys are level indices. Each value is a
            flat NumPy array containing all collected loss values for that
            combination.
    """
    losses_vs_learning_levels = {
        loss: {level: [] for level in levels} for loss in losses
    }

    for d in datapoints:
        for prefix in prefixes:
            d[f"{prefix}_sum"] = sum(
                v for k, v in d.items()
                if k.startswith(prefix)
            )
        d_level = d[level_key]
        for loss in losses:
            for level in levels:
                mask = d_level == level
                if filter_out_spaces:
                    tokens = d["tokens"][1:-1]  # remove BOS and EOS tokens
                    space_mask = (
                        tokens != 32
                    )  # remove spaces (32 is the ascii code for space)
                    mask = np.logical_and(mask, space_mask)
                if mask.sum() > 0:
                    losses_len = len(d[loss])
                    losses_vs_learning_levels[loss][level].append(d[loss][mask[:losses_len]])

    for loss in losses:
        for level in levels:
            c = np.concatenate(losses_vs_learning_levels[loss][level])
            losses_vs_learning_levels[loss][level] = c

    return losses_vs_learning_levels

def eval_model(checkpoint_path, losses, levels, num_datapoints=1000, prefixes=None):
    config_file = checkpoint_path + "/config.yaml"
    cfg = OmegaConf.load(config_file)
    if cfg.train_whole_model:
        print("load full model")
        cfg.checkpointer.checkpoint_dir = checkpoint_path
        cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    else:
        cfg.checkpointer._self_prediction_checkpoint_dir = checkpoint_path
        cfg.checkpointer._self_prediction_checkpoint = "meta_model_None.pt"
        
    cfg.checkpointer.output_dir = "/home/suurajperpeli/models/pfa_eval"
    cfg.train_from_scratch = False
    cfg.compile = False
    cfg.metric_logger._component_ = "torchtune.training.metric_logging.DiskLogger"
    recipe = SelfPredictionTrainingRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)

    kwargs = dict(recipe.cfg.dataset)
    kwargs.pop("_component_")
    kwargs["tokenizer"] = recipe._tokenizer
    dataset = learning_levels_pfa_dataset(**kwargs)

    recipe._model.eval()
    datapoints = process_data(recipe,num_datapoints=num_datapoints,dataset=dataset,batch_size=cfg.batch_size,)
    

    lvll = compute_losses_per_level(datapoints, 
                                    losses=losses, 
                                    levels=levels,
                                    filter_out_spaces=False,
                                    prefixes=prefixes,)
    lvlls = compute_losses_per_level_statistics(lvll,
                                                losses=losses,
                                                levels=levels,)
    for k, v in lvlls.items():
        for l, s in v.items():
            for f, value in s.items():
                if type(value) != int:
                    lvlls[k][l][f] = float(value)
    return datapoints, lvlls

if __name__ == "__main__":

    levels = (0,1,2,3,4)
    num_datapoints = 1000
    num_quantizers = 2

    losses = ['next_token_losses', 'phi_losses_sum', 'latent_entropy_sum', 'latent_losses_sum']
    prefixes = ['phi_losses', 'latent_losses', 'latent_entropy']

    api = wandb.Api()
    runs = api.runs("hidden-state-predictions/llama-0.1B-gumbel-layers")
    layers_vs_stats = []
    for run in runs:

        #applying a filter for layer 10 and llf 1e-2
        if not 'layer-10-llf-1e-2' in run.name:
            continue
        print(f"Evaluating run: {run.name} with id: {run.id}")
        
        checkpoint_base = "/home/suurajperpeli/llama-runs/runs/learning_levels_sweep_"  # llama-0.1B
        checkpoint_path = checkpoint_base + run.id
        datapoints, stats = eval_model(checkpoint_path, losses, levels, num_datapoints=num_datapoints, prefixes=prefixes)
        layers_vs_stats.append({
            'layer': int(run.name.split('-')[3]),
            'seed': int(run.name.split('-')[1]),
            'statistics': stats, 
            'run_id': str(run.id)} )
    with open(f"/home/suurajperpeli/thesis-evaluations/pfa_evals_0.1B/rq-layer-10-llf-1e-2-run.json", 'w') as f:
        print('saving run')
        json.dump(layers_vs_stats, f)
