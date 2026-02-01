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
import seaborn as sns
import pandas as pd
import wandb


from dataset_classes import learning_levels_pfa_dataset, PackedOnTheFlyDataset
from evaluation.pfa_evaluation import generate_per_token_losses, decode_token_by_token, process_data, compute_losses_per_level, compute_losses_per_level_statistics
from training import SelfPredictionTrainingRecipeDistributed

def eval_model(checkpoint_path, losses, levels, num_datapoints=1000):
    config_file = checkpoint_path + "/config.yaml"
    try:
        cfg = OmegaConf.load(config_file)
    except:
        return None
    if cfg.train_whole_model:
        print("load full model")
        cfg.checkpointer.checkpoint_dir = checkpoint_path
        cfg.checkpointer.checkpoint_files = ["torchtune_model_last.pt"]
    else:
        cfg.checkpointer._self_prediction_checkpoint_dir = checkpoint_path
        cfg.checkpointer._self_prediction_checkpoint = "meta_model_None.pt"
        
    cfg.checkpointer.output_dir = "/home/woody/iwi5/iwi5368h/models/pfa_eval"
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
                                    filter_out_spaces=False,)
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

    losses = ['next_token_losses', 'phi_losses0', 'latent_entropy0', 'latent_losses0']
    levels = (0,1,2,3,4)
    num_datapoints = 1000

    api = wandb.Api()
    runs = api.runs("hidden-state-predictions/llama-0.1B-gumbel-layers")
    layers_vs_stats = []
    for run in runs:
        checkpoint_base = "/home/woody/iwi5/iwi5368h/models/llama_0.1B_PHi/learning_levels_sweep_"  # llama-0.1B
        checkpoint_path = checkpoint_base + run.id
        datapoints, stats = eval_model(checkpoint_path, losses, levels)
        layers_vs_stats.append({
            'layer': int(run.name.split('-')[3]),
            'seed': int(run.name.split('-')[1]),
            'statistics': stats, 
            'run_id': str(run.id)} )
    with open(f"/home/woody/iwi5/iwi5368h/evaluations/pfa_evals_0.1B/layers-run-gumbel.json", 'w') as f:
        print('saving run')
        json.dump(layers_vs_stats, f)
