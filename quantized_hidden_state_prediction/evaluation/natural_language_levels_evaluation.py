import numpy as np
import plotly.graph_objects as go
import torch

from torchtune.data import padded_collate_packed
from torchtune.modules import delete_kv_caches

from dataset_classes import (
    PackedOnTheFlyDataset,
    languages_levels_eval_dataset,
)
from evaluation.pfa_evaluation import (
    compute_losses_per_level,
    compute_losses_per_level_statistics,
    generate_per_token_losses,
)
from utils.misc import seed_everything


LEVEL_NAME_TO_INDEX = {
    "predictable": 0,
    "licenses": 1,
    "shuffled": 2,
    "literature": 3,
    "code": 4,
}


@torch.no_grad()
def process_data(
    recipe,
    num_datapoints=10,
    dataset=None,
    batch_size=4,
):
    """
    Processes a dataset to generate per-token losses for each sample.

    This takes a dataset and runs it through the model provided in the recipe to
    compute various per-token losses (e.g., next-token prediction, hidden state
    prediction). The batched results are then unpacked and reorganized into a
    list of individual data points, with all losses aligned to their
    corresponding tokens.

    Args:
        recipe (Any): A recipe object containing the model (`_model`) and a
            default dataloader (`_dataloader`).
        num_datapoints (int, optional): The target number of individual data points
            to process and return. Defaults to 10.
        dataset (Optional[torch.utils.data.Dataset], optional): The dataset to process.
            If None, the dataset from `recipe._dataloader` is used. Defaults to None.
        batch_size (int, optional): The number of samples to process in each batch.
            Defaults to 4.

    Returns:
        List[Dict[str, np.ndarray]]: A list of dictionaries, where each dictionary
            represents a single data point. Each dictionary contains the token sequence
            and various corresponding per-token loss arrays as NumPy arrays.
    """
    if dataset is None:
        dataset = recipe._dataloader.dataset
    if not hasattr(dataset, "max_sample_length"):
        max_sample_length = 2048
    else:
        max_sample_length = dataset.max_sample_length
    packed_dataset = PackedOnTheFlyDataset(dataset,
                                           max_seq_len=max_sample_length)
    datapoints = []
    finished = False
    for idx in range(0, len(dataset), batch_size):
        print(f"Processing batch {len(datapoints) + 1}/{num_datapoints}")
        batch = []
        for _ in range(batch_size):
            try:
                sample = next(packed_dataset)
            except StopIteration:
                finished = True
                break
            batch.append(sample)
        if len(batch) == 0 and finished:
            break

        model_batch = padded_collate_packed(batch)

        # Calculate per-token losses
        per_token_losses = generate_per_token_losses(recipe, model_batch)
        if "next_prediction_mse" in per_token_losses:
            del per_token_losses["next_prediction_loss"]  # we do not need this loss

        for key, value in per_token_losses.items():
            if 'next' in key:
                per_token_losses[key] = torch.cat([torch.zeros_like(value[:, 0:1]), value], dim=1)
            if key == 'next_token_losses':
                per_token_losses[key] = per_token_losses[key][:, :-1]

        # split into datapoints
        current_datapoints = []
        for b in range(len(batch)):
            start_idx = 0
            sample = batch[b]
            for i, seq_len in enumerate(sample['seq_lens']):
                if sample['tokens'][start_idx] == 0 and i == len(sample['seq_lens']) - 1:
                    # this means that it's the last sequence and it's only padding
                    continue
                end_idx = start_idx + seq_len
                current_datapoint = {}
                for key, value in sample.items():
                    if len(value) != len(sample['tokens']):
                        if 'debug' in key:
                            continue
                        if key[0] == '_':
                            key = key[1:]
                        current_datapoint[key] = value[i]
                    else:
                        current_datapoint[key] = value[start_idx:end_idx].detach().cpu().numpy()
                for key, value in per_token_losses.items():
                    if key == 'tokens':
                        continue
                    if type(value) == torch.Tensor and value.numel() == 1:
                        # this means it's a global loss for training, not a token-wise loss
                        continue
                    value = value[b][start_idx:end_idx]
                    if type(value) is torch.Tensor:
                        value = value.detach().cpu().float().numpy()
                    current_datapoint[key] = value

                # make sure that invalid tokens have 0 loss
                for key, value in current_datapoint.items():
                    if 'next' in key:
                        current_datapoint[key][0] = 0.

                if '_level' in sample:
                    if type(sample["_level"]) == list and type(sample["_level"][0]) == str:
                        level = LEVEL_NAME_TO_INDEX[sample['_level'][i]]
                    else:
                        level = sample["_level"][i]
                    current_datapoint["level"] = np.ones(current_datapoint["next_token_losses"].shape[0],
                                                             dtype=int) * level

                # valid datapoint only if not all tokens are 0
                is_valid_datapoint = not np.all(current_datapoint['tokens'] == 0)
                if is_valid_datapoint:
                    current_datapoints.append(current_datapoint)
                start_idx = end_idx
        datapoints.extend(current_datapoints)
        if len(datapoints) >= num_datapoints:
            break
        if finished:
            break
    if len(datapoints) > num_datapoints:
        datapoints = datapoints[:num_datapoints]
    return datapoints


@torch.no_grad()
def nl_learning_levels_evaluation(recipe, num_datapoints=500):
    """
    Evaluates a model on various natural language "learning levels."

    This function processes a specialized evaluation dataset where samples are
    categorized by complexity (e.g., "predictable," "code," "literature").
    It computes per-token losses for each category, generates bar charts
    comparing these losses, and calculates an "interestingness ratio."
    This ratio compares the model's primary loss (e.g., PHi loss) on
    "interesting" tasks versus "uninteresting" tasks.

    The model is returned to training mode upon completion.

    Args:
        recipe (Any): A recipe object containing the model (`_model`), tokenizer
            (`_tokenizer`), and training configuration (`cfg`).
        num_datapoints (int, optional): The number of data points to process for
            the evaluation. Defaults to 500.

    Returns:
        Tuple[Dict[str, go.Figure], Dict[str, float]]: A tuple containing:
        - A dictionary where keys are loss names (str) and values are the
          corresponding Plotly bar chart figures.
        - A dictionary containing the calculated "interestingness_ratio" (float).
    """
    recipe._model.eval()
    dataset = languages_levels_eval_dataset(recipe._tokenizer, 
                                            location='/home/hpc/iwi5/iwi5368h/predicting-hidden-states/data/natural_language_levels',)
    datapoints = process_data(
        recipe,
        num_datapoints=num_datapoints,
        dataset=dataset,
        batch_size=recipe.cfg.batch_size,
    )

    losses = ["next_token_losses"]
    interestingness_criterion = "next_token_losses"
    if recipe._model.self_prediction_layer.quantizer is not None:
        for i in range(recipe._model.self_prediction_layer.quantizer.num_quantizers):
            losses.append(f"phi_losses{i}",)
            losses.append(f"latent_losses{i}",)
            losses.append(f"latent_entropy{i}",)
    else:
        losses.append("phi_losses")
        losses.append("latent_losses")
        losses.append("latent_entropy")

    levels = (0, 1, 2, 3, 4)
    interesting_levels = (3, 4)
    uninteresting_levels = (0, 1, 2)
    losses_vs_learning_levels = compute_losses_per_level(
        datapoints,
        filter_out_spaces=False,
        losses=losses,
        levels=levels,
        level_key='level',
    )
    losses_vs_learning_levels_statistics = compute_losses_per_level_statistics(
        losses_vs_learning_levels,
        losses=losses,
        levels=levels
    )

    plotly_figure_dict = {}
    interestingness_ratio = 0.0
    index_to_level_name = {v: k for k, v in LEVEL_NAME_TO_INDEX.items()}
    for i, loss in enumerate(losses):
        means = [
            losses_vs_learning_levels_statistics[loss][level]["mean"]
            for level in levels
        ]
        if loss == interestingness_criterion:
            interesting_means = [means[level] for level in interesting_levels]
            uninteresting_means = [means[level] for level in uninteresting_levels]
            interestingness_ratio = np.mean(interesting_means) / np.mean(uninteresting_means)
        fig = go.Figure(data=[go.Bar(x=levels, y=means)])
        fig.update_layout(
            title=f"{loss.capitalize()}",
            xaxis_title="Learning level",
            yaxis_title="",
            xaxis=dict(tickvals=levels, ticktext=[index_to_level_name[l] for l in levels])
        )
        plotly_figure_dict[loss] = fig
    recipe._model.train()
    return plotly_figure_dict, {'interestingness_ratio': interestingness_ratio}