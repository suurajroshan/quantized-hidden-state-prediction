# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from dataclasses import dataclass

import os
import matplotlib
import matplotlib.colors as mcolors

# import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from scipy.special import comb

# from tests.torchtune.dataset_classes.test_learning_levels_pfa_dataset import learning_levels_languages_dataset
from dataset_classes import learning_levels_pfa_dataset
# from plotly.subplots import make_subplots
from torchtune import utils
from torchtune.modules import delete_kv_caches

from torchtune.data import padded_collate_packed
from dataset_classes import PackedOnTheFlyDataset
from models.tokenizer import ASCIITokenizer
from evaluation.custom_generation import generate, generate_next_token_only_lowercase
from utils.misc import bootstrapped_mean_and_ci

import pickle

log = utils.get_logger("DEBUG")


def decode_token_by_token(tokens, tokenizer):
    """
    Decodes a tensor of tokens individually.

    Handles `ASCIITokenizer` separately for direct character conversion.
    For other tokenizers, it decodes tokens one by one, using a duplication
    strategy (e.g., decoding [T,T]) to isolate the string representation of
    each token, which can be useful if single token decoding has artifacts.

    Args:
        tokens (torch.Tensor): 1D or 2D tensor of token IDs.
        tokenizer: Tokenizer with a `decode` method.

    Returns:
        list: List of decoded token strings. If input is 2D, returns a list of lists.
    """

    # Special handling for ASCIITokenizer for simple char-by-char decoding.
    if type(tokenizer) is ASCIITokenizer:
        string = tokenizer.decode(tokens.squeeze().tolist())
        tokens_by_tokens = [c for c in string]
        return tokens_by_tokens

    # Ensure tokens are batched (2D tensor).
    if len(tokens.shape) == 1:
        tokens_batch = tokens.unsqueeze(0)
        input_was_1d = True
    else:
        tokens_batch = tokens
        input_was_1d = False

    decoded_batch = []
    for current_tokens_sequence in tokens_batch:
        # Decode pairs of identical tokens: [[t1,t1], [t2,t2], ...].
        # This helps isolate the true representation of a single token if the
        # tokenizer adds prefixes/suffixes when decoding individual tokens.
        tokens_duplicate_list = [[t.item(), t.item()] for t in current_tokens_sequence.squeeze()]
        tokens_duplicate_decoded = [
            tokenizer.decode(pair) for pair in tokens_duplicate_list
        ]

        tokens_by_tokens_for_sequence = []
        for s in tokens_duplicate_decoded:
            length_of_single_token_representation = len(s) - (len(s) // 2)  # ceil(len(s)/2)
            tokens_by_tokens_for_sequence.append(s[-length_of_single_token_representation:])
        decoded_batch.append(tokens_by_tokens_for_sequence)

    # Return a single list if input was 1D, otherwise the batch of lists.
    if input_was_1d:
        return decoded_batch[0]
    else:
        return decoded_batch


def generate_per_token_losses(recipe, batch):
    """
    Calculates per-token losses for a given batch of data using a specified recipe.

    This function processes a batch to compute the cross-entropy loss for each
    token in the input sequences. It also decodes the input tokens and incorporates
    any additional losses recorded in the model (e.g., self-prediction losses).

    Args:
        recipe: A recipe object containing the model (`_model`), tokenizer (`_tokenizer`),
                device (`_device`), loss function (`_loss_fn`), and
                `ignore_labels_cache`.
        batch (dict): A dictionary containing the input data, typically including
                      'tokens' and 'labels'. 'labels' will be removed from the batch.

    Returns:
        dict: A dictionary containing per-token loss information:
              - 'tokens': The original input token IDs.
              - 'decoded_tokens': The string representation of each input token.
              - 'next_token_losses': The cross-entropy loss for predicting the next token.
              - Additional keys for other losses, derived from `recipe._model.self_prediction_losses`.
    """
    utils.batch_to_device(batch, recipe._device)
    labels = batch.pop("labels")
    labels = torch.hstack(  # shift labels by one (for next token prediction)
        (labels[..., 1:], recipe.ignore_labels_cache[: labels.shape[0]])
    )
    logits = recipe._model(**batch)
    if type(logits) is list:
        logits = torch.cat(logits, dim=1)
    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=recipe._loss_fn.ignore_index,
    ).view_as(labels)
    decoded_tokens = decode_token_by_token(batch["tokens"], tokenizer=recipe._tokenizer)
    per_token_losses = {
        "tokens": batch["tokens"],
        "decoded_tokens": decoded_tokens,
        "next_token_losses": losses,
    }
    additional_losses = recipe._model.self_prediction_losses.losses
    recipe._model.self_prediction_losses.reset()

    for key, value in additional_losses.items():
        new_key = key.replace("tokenwise ", "").replace(" ", "_")
        per_token_losses[new_key] = value

    return per_token_losses


def get_language_complexity(vocab_size, num_edges, num_states, total_tokens):
    """
    Calculates the theoretical complexity of defining a Probabilistic Finite Automaton (PFA).

    This complexity is based on the formula for encoding a PFA with 'num_states' (n) states,
    'num_edges' (m) edges, and a vocabulary of 'vocab_size' (v) chosen from
    'total_tokens' (V) available tokens. See associated research paper for details
    on the formula: C(A) = log(C(V,v)) + m * (2*log(n) + log(v)).

    Note: The function uses natural logarithm (np.log). For complexity in bits,
    log base 2 (np.log2) would typically be used as in the reference paper's formula.

    Args:
        vocab_size (int): The number of unique tokens (v) used by the PFA.
        num_edges (int): The number of edges (m) in the PFA.
        num_states (int): The number of states (n) in the PFA.
        total_tokens (int): The total number of tokens (V) in the overall vocabulary
                            from which the PFA's vocabulary is chosen.

    Returns:
        float: The calculated complexity value.
    """

    # Calculate the "cost" of specifying the PFA's vocabulary: log(C(V, v)).
    # This is the information needed to choose 'v' tokens out of 'V' total_tokens.
    # comb(total_tokens, vocab_size) computes "total_tokens choose vocab_size".
    vocab_bits = np.log(comb(total_tokens, vocab_size))

    # Calculate the "cost" of specifying all edges: m * (2*log(n) + log(v)).
    # For each of the 'num_edges' (m):
    #   - log(num_states) for the origin state (n options).
    #   - log(num_states) for the target state (n options).
    #   - log(vocab_size) for the token label on the edge (v options).
    edge_bits = num_edges * (2 * np.log(num_states) + np.log(vocab_size))

    # Total complexity is the sum of vocabulary specification and edge specification.
    return vocab_bits + edge_bits


# no gradients for this function
@torch.no_grad()
def process_data(
    recipe,
    num_datapoints=10,
    dataset=None,
    batch_size=4,
):
    """
    Processes data from a dataset to generate a specified number of datapoints
    with per-token loss information.

    This function iterates through the dataset, forms batches, calculates
    per-token losses using the `generate_per_token_losses` function,
    adjusts losses related to hidden state predictions by padding,
    and then splits the batch results into individual datapoint dictionaries.

    Args:
        recipe: A recipe object containing model, tokenizer, dataloader (for default dataset), etc.
        num_datapoints (int, optional): The target number of datapoints to generate. Defaults to 10.
        dataset (Dataset, optional): The dataset to process. If None, uses `recipe._dataloader.dataset`.
                                     Defaults to None.
        batch_size (int, optional): The number of samples to process in each batch. Defaults to 4.

    Returns:
        list: A list of dictionaries, where each dictionary represents a datapoint
              containing tokens, decoded tokens, various per-token losses, and other
              relevant information from the original sample, all as numpy arrays.
    """
    if dataset is None:
        dataset = recipe._dataloader.dataset

    packed_dataset = PackedOnTheFlyDataset(dataset,
                                           max_seq_len=dataset.max_sample_length)
    datapoints = []
    finished_processing_dataset = False

    # Iterate through the dataset in batches.
    for idx in range(0, len(dataset), batch_size):
        print(f"Processing batch {len(datapoints)+1}/{num_datapoints}")
        current_batch_samples = []
        for i in range(batch_size):
            # get next item from the packed_dataset
            try:
                sample = next(packed_dataset)
                current_batch_samples.append(sample)
            except StopIteration:
                finished_processing_dataset = True
                break
        model_batch = padded_collate_packed(current_batch_samples)

        # Calculate per-token losses
        per_token_losses = generate_per_token_losses(recipe,
                                                     model_batch)

        for key, value in per_token_losses.items():
            if 'next' in key:
                per_token_losses[key] = torch.cat([torch.zeros_like(value[:, 0:1]), value], dim=1)
            if key == 'next_token_losses':
                per_token_losses[key] = per_token_losses[key][:, :-1]


        # split into datapoints
        current_datapoints = []
        for b in range(len(current_batch_samples)):
            start_idx = 0
            sample = current_batch_samples[b]
            for i, seq_len in enumerate(sample['seq_lens']):
                end_idx = start_idx + seq_len
                current_datapoint = {}
                for key, value in sample.items():
                    if len(value) != len(sample['tokens']):
                        continue
                    current_datapoint[key] = value[start_idx:end_idx].detach().cpu().numpy()
                for key, value in per_token_losses.items():
                    if key == 'tokens':
                        continue
                    if type(value) != torch.Tensor or value.numel() <= 1:
                        continue
                    value = value[b][start_idx:end_idx]
                    if type(value) is torch.Tensor:
                        value = value.detach().cpu().float().numpy()
                    current_datapoint[key] = value

                # valid datapoint only if not all tokens are 0
                is_valid_datapoint = not np.all(current_datapoint['tokens'] == 0)
                if is_valid_datapoint:
                    current_datapoints.append(current_datapoint)
                start_idx = end_idx
        datapoints.extend(current_datapoints)
        if len(datapoints) >= num_datapoints:
            break
        if finished_processing_dataset:
            break
    if len(datapoints) > num_datapoints:
        datapoints = datapoints[:num_datapoints]
    return datapoints


def compute_losses_per_level(
    datapoints,
    losses=("next_token_losses", "phi_losses"),
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


def compute_losses_per_level_statistics(
    losses_vs_learning_levels,
    losses=("next_token_losses", "phi_losses"),
    levels=(0, 1, 2, 3, 4),
):
    """
    Computes descriptive statistics for aggregated per-level losses.

    This function takes a dictionary of loss arrays (typically the output from
    `compute_losses_per_level`) and calculates the mean, median, standard
    deviation, count, and standard error for each loss type at each level.

    Args:
        losses_vs_learning_levels (Dict[str, Dict[int, np.ndarray]]):
            A nested dictionary where keys are loss names and level indices,
            and values are NumPy arrays of the corresponding loss values.
        losses (Tuple[str, ...], optional): A tuple of strings specifying which
            loss keys to process from the input dictionary.
            Defaults to ("next_token_losses", "phi_losses").
        levels (Tuple[int, ...], optional): A tuple of the integer level indices
            to process. Defaults to (0, 1, 2, 3, 4).

    Returns:
        Dict[str, Dict[int, Dict[str, float]]]: A nested dictionary containing
            the calculated statistics. The structure is:
            `{loss_name: {level: {"mean": ..., "median": ..., "std": ..., "num": ..., "std_err": ...}}}`
    """
    losses_vs_learning_levels_statistics = {
        loss: {level: {} for level in levels} for loss in losses
    }
    for loss in losses:
        for level in levels:
            c = losses_vs_learning_levels[loss][level]
            losses_vs_learning_levels_statistics[loss][level]["mean"] = c.mean()
            losses_vs_learning_levels_statistics[loss][level]["median"] = np.median(c)
            losses_vs_learning_levels_statistics[loss][level]["std"] = c.std()
            losses_vs_learning_levels_statistics[loss][level]["num"] = len(c)
            losses_vs_learning_levels_statistics[loss][level]["std_err"] = c.std() / (
                len(c) ** 0.5
            )
    return losses_vs_learning_levels_statistics


def compute_bins(values, num_bins=20):
    """
    Calculates bin edges for quantile-based binning.

    This function divides a set of values into bins containing an approximately
    equal number of data points. It sorts the values and picks bin edges
    such that each bin represents a quantile of the data.

    Args:
        values (np.ndarray): A 1D NumPy array of values to be binned.
        num_bins (int, optional): The desired number of bins. Defaults to 20.

    Returns:
        np.ndarray: A NumPy array of bin edges of length `num_bins + 1`.
    """
    sorted_values = np.sort(values)
    num_values_per_bin = len(sorted_values) // num_bins
    bins = []
    for i in range(num_bins):
        bins.append(sorted_values[i * num_values_per_bin])
    bins.append(sorted_values[-1])
    bins = np.array(bins)
    return bins


def compute_controlled_losses(
    losses_vs_learning_levels,
    control_loss="next_token_losses",
    num_bins=20,
    per_level_binning=False,
    losses=("next_token_losses", "phi_losses"),
    levels=(0, 1, 2, 3, 4),
):
    """
    Re-aggregates losses by binning them against a specified control loss.

    This function is used to analyze the behavior of certain losses (e.g., PHi loss)
    while controlling for the effect of another loss (e.g., next-token loss).
    It groups data points into bins based on their `control_loss` value and then
    computes statistics for all specified `losses` within each bin.

    Args:
        losses_vs_learning_levels (Dict[str, Dict[int, np.ndarray]]):
            A nested dictionary of aggregated losses, typically the output of
            `compute_losses_per_level`.
        control_loss (str, optional): The key for the loss variable to use for
            binning. Defaults to "next_token_losses".
        num_bins (int, optional): The number of bins to create. Defaults to 20.
        per_level_binning (bool, optional): If True, bin edges are calculated
            independently for each level. If False, a single set of global bin
            edges is calculated from all data across all levels. Defaults to False.
        losses (Tuple[str, ...], optional): A tuple of loss names to analyze.
            Defaults to ("next_token_losses", "phi_losses").
        levels (Tuple[int, ...], optional): A tuple of level indices to process.
            Defaults to (0, 1, 2, 3, 4).

    Returns:
        Tuple[Dict, Dict]: A tuple containing two dictionaries:
        - `controlled_losses`: A nested dictionary with the raw loss values
          grouped by loss type, level, and control loss bin.
          Structure: `{loss: {level: {bin: np.ndarray}}}`
        - `controlled_losses_statistics`: A nested dictionary with descriptive
          statistics (mean, std, num, std_err) for each group.
          Structure: `{loss: {level: {bin: {statistic: value}}}}`
    """
    if not per_level_binning:
        all_controls = []
        for level in levels:
            all_controls.append(losses_vs_learning_levels[control_loss][level])
        all_controls = np.concatenate(all_controls)
        bins = compute_bins(all_controls, num_bins)

    controlled_losses = {
        loss: {level: {} for level in levels} for loss in losses
    }  # each loss for each level, in bins of the control loss

    for level in levels:
        if per_level_binning:
            controll = losses_vs_learning_levels[control_loss][level]
            bins = compute_bins(controll, num_bins)
        controll = losses_vs_learning_levels[control_loss][level]
        bin_assignment = np.digitize(controll, bins) - 1
        for loss in losses:
            for bin in range(num_bins):
                is_in_bin = bin_assignment == bin
                controlled_losses[loss][level][bin] = losses_vs_learning_levels[loss][
                    level
                ][is_in_bin]

    controlled_losses_statistics = {
        loss: {level: {} for level in levels} for loss in losses
    }
    for loss in losses:
        for level in levels:
            for bin in range(num_bins):
                c = controlled_losses[loss][level][bin]
                controlled_losses_statistics[loss][level][bin] = {
                    "mean": c.mean(),
                    "std": c.std(),
                    "num": len(c),
                    "std_err": c.std() / (len(c) ** 0.5),
                }

    return controlled_losses, controlled_losses_statistics


def recognized_prefix_length(M: np.ndarray, tokens: list[int]) -> (int, bool):
    """
    Simulates a Non-deterministic Finite Automaton (NFA) to find the longest
    prefix of a token sequence that it recognizes starting from state 0.

    The function tracks the set of all possible states the NFA could be in at
    each step. A prefix is recognized as long as there is at least one valid
    path from the set of current states to a next state using the current token.

    Args:
      - M: an n x n transition matrix (NumPy array).
           M[p, q] = -1 indicates no transition from state p to q.
           M[p, q] = t >= 0 indicates a transition from p to q emitting token t.
      - tokens: The input sequence (list of integers).

    Returns:
      - prefix_len: The length of the longest prefix recognized by the automaton.
      - full_match: True if the entire input sequence is recognized, False otherwise.
    """

    n = M.shape[0]  # number of states
    # Q_i will be stored as a NumPy 1D array of valid states
    Q_current = np.arange(n)  # We can start from any state: Q_0 = {0, 1, ..., n-1}

    # We will track how many tokens we've successfully consumed
    consumed = 0

    for token in tokens:
        # For each state p in Q_current, we check if there's a q with M[p, q] == token
        next_states = []

        for p in Q_current:
            # Find all q such that M[p,q] == token
            # np.where(...) returns tuple (array_of_indices,), so we take [0]
            q_candidates = np.where(M[p, :] == token)[0]
            next_states.extend(q_candidates)

        # Keep unique states (we don't want duplicates)
        next_states = np.unique(next_states)

        if len(next_states) == 0:
            # No valid transition for this token
            return consumed, False

        # Update Q_current for the next iteration
        Q_current = next_states
        consumed += 1  # We matched one more token

    # If we exit the loop without breaks, we matched all tokens
    return consumed, True


@torch.no_grad()
def evaluate_language_generation_length(recipe,
                                        num_samples=1000,
                                        batch_size=100):
    """
    Evaluates a model's ability to generate syntactically correct PFA sequences.

    This function tests how well a model can perform in-context learning of a
    formal language. For each sample, the model is prompted with several example
    sequences from a randomly generated Probabilistic Finite Automaton (PFA). It is
    then tasked with generating a new sequence from that same PFA.

    The evaluation measures the length of the longest prefix of the newly
    generated sequence that is syntactically valid according to the PFA's rules.

    Args:
        recipe (Any): A recipe object containing the model (`_model`), tokenizer
            (`_tokenizer`), and configuration (`cfg`).
        num_samples (int, optional): The total number of generation tasks to
            evaluate. Defaults to 1000.
        batch_size (int, optional): The number of samples to process in each batch.
            Defaults to 100.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results, including:
            - "accepted_lengths": A list of the correctly generated prefix lengths.
            - "mean": The mean of the accepted lengths.
            - "std": The standard deviation.
            - "std_err": The standard error of the mean.
            - "ci_mean": The mean from bootstrapping.
            - "ci_95": The 95% confidence interval for the mean.
    """
    kwargs = dict(recipe.cfg.dataset)
    kwargs.pop("_component_")
    kwargs["tokenizer"] = recipe._tokenizer
    kwargs["word_perturbation_rate"] = 0.
    kwargs["token_perturbation_rate"] = 0.
    kwargs["sequences_per_language_min"] = 10
    kwargs["sequences_per_language_min"] = 11
    dataset = learning_levels_pfa_dataset(**kwargs)

    char_to_idx = {c: i for i, c in enumerate(dataset.letters)}

    with recipe._device:
        recipe._model.setup_caches(
            batch_size=batch_size,
            dtype=recipe._dtype,
            decoder_max_seq_len=2048,
        )
    previous_num_chunks = recipe._model.num_output_chunks
    recipe._model.num_output_chunks = 0

    correct_lengths = []
    for i in range(0, num_samples, batch_size):
        recipe._model.reset_caches()
        recipe._model.self_prediction_losses.reset()
        tokens = []
        transition_symbols = []
        for j in range(batch_size):
            sample = dataset[i]
            tokens.append(sample['tokens'])
            transition_symbols.append(np.array(sample['_transition_symbols'][0]))
        generated_sequences = generate(tokens,
                     recipe=recipe,
                     max_new_tokens=101,
                     top_k=18,
                     stop_tokens=[32],
                     custom_generate_next_token=generate_next_token_only_lowercase)
        for j, d in enumerate(generated_sequences):
            decoded_tokens = d['decoded_tokens']

            sequences = decoded_tokens.split(' ')
            sequences_np = [np.array([char_to_idx[c] if c in char_to_idx else 25 for c in s]) for s in sequences]
            current_symbols = transition_symbols[j]

            for seq_i, seq in enumerate(sequences_np):
                len_correct, accepted = recognized_prefix_length(current_symbols, seq)
                if seq_i == len(sequences_np) - 1:
                    # print(seq)
                    # print(f"{i + j + 1}/{num_samples}: {len_correct}/{len(seq)} correct")
                    correct_lengths.append(len_correct)
                else:
                    assert accepted

    correct_lengths = np.array(correct_lengths)
    ci_mean, ci_lower, ci_higher = bootstrapped_mean_and_ci(correct_lengths, num_samples=10000)

    delete_kv_caches(recipe._model)
    recipe._model.num_output_chunks = previous_num_chunks

    return {
        "accepted_lengths": correct_lengths.tolist(),
        "mean": correct_lengths.mean(),
        "std": correct_lengths.std(),
        "std_err": correct_lengths.std() / np.sqrt(correct_lengths.size),
        "ci_mean": ci_mean,
        "ci_95": [ci_lower, ci_higher]
    }


def pfa_training_evaluation(recipe,
                            num_datapoints=500,
                            ic_generalization_evaluation=True):
    """
    Evaluates a model on PFA-based learning tasks.

    This function assesses a model's in-context learning capabilities by processing
    data from `learning_levels_pfa_dataset`. It calculates per-token losses across
    various task complexities (e.g., memorization, generalization, random sequences),
    generates summary plots, and computes key metrics.

    The primary metric is the "interestingness ratio," which compares the model's
    loss on complex, generative tasks versus simple, memorization-based tasks.
    Optionally, it can also evaluate the model's ability to generate syntactically
    correct sequences.

    Args:
        recipe (Any): A recipe object containing the model (`_model`), tokenizer,
            and configuration.
        num_datapoints (int, optional): The number of data points to process for
            calculating losses. Defaults to 500.
        ic_generalization_evaluation (bool, optional): If True, runs an additional
            evaluation to measure the length of correctly generated PFA sequences.
            Defaults to True.

    Returns:
        Tuple[Dict[str, go.Figure], Dict[str, float]]: A tuple containing:
        - A dictionary mapping loss names (e.g., "phi_losses") to Plotly bar
          chart figures visualizing the loss per learning level.
        - A dictionary of scalar evaluation metrics, including
          "interestingness_ratio" and optionally "length_generalization".
    """
    recipe._model.eval()

    kwargs = dict(recipe.cfg.dataset)
    kwargs.pop("_component_")
    kwargs["tokenizer"] = recipe._tokenizer
    dataset = learning_levels_pfa_dataset(**kwargs)

    datapoints = process_data(
        recipe,
        dataset=dataset,
        num_datapoints=num_datapoints,
    )
    # with open(os.path.join(recipe.pickle_dir, "eval_metrics.pkl"), "wb") as f:
    #     pickle.dump({recipe.global_step: datapoints}, f)

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
    
    log.info(datapoints[0].keys())
    
    # if "phi_losses" in datapoints[0]:
    #     losses.append("phi_losses")
    #     interestingness_criterion = "phi_losses"

    level_names = [
        "memorized sequence",
        "memorized language",
        "learned vocabulary",
        "learned language",
        "random",
        "copy",
    ]
    levels = tuple(dataset.included_learning_levels)
    interesting_levels = (2, 3)
    uninteresting_levels = (0, 1, 4, 5)
    # find the intersection between both interesting and uninteresting levels and the levels in the dataset
    interesting_levels = list(set(interesting_levels).intersection(set(levels)))
    uninteresting_levels = list(set(uninteresting_levels).intersection(set(levels)))

    losses_vs_learning_levels = compute_losses_per_level(
        datapoints,
        filter_out_spaces=False,
        losses=losses,
        levels=levels
    )
    losses_vs_learning_levels_statistics = compute_losses_per_level_statistics(
        losses_vs_learning_levels,
        losses=losses,
        levels=levels
    )

    if ic_generalization_evaluation:
        length_generalization_results = evaluate_language_generation_length(recipe, num_samples=500)
        length_generalization = length_generalization_results['mean']
    else:
        length_generalization = None

    plotly_figure_dict = {}
    interestingness_ratio = 0.0

    group_losses = [[]]
    # group_losses = [
    #     ["prediction_entropy", "target_entropy"],
    #     ]

    loss_filter = [False for _ in losses]
    for i, loss in enumerate(losses):
        for group in group_losses:
            if loss in group:
                loss_filter[i] = True

    for i, loss in enumerate(losses):
        if loss_filter[i]:
            continue
        means = [
            losses_vs_learning_levels_statistics[loss][level]["mean"]
            for level in levels
        ]
        if loss == interestingness_criterion:
            if len(interesting_levels) > 0 and len(uninteresting_levels) > 0:
                interesting_means = [means[level] for level in interesting_levels]
                uninteresting_means = [means[level] for level in uninteresting_levels]
                interestingness_ratio = np.mean(interesting_means) / np.mean(
                    uninteresting_means
                )
        fig = go.Figure(data=[go.Bar(x=levels, y=means)])
        # add level names
        fig.update_layout(
            title=f"{loss.capitalize()}",
            xaxis_title="Learning level",
            yaxis_title="",
            xaxis=dict(tickvals=levels, ticktext=[level_names[l] for l in levels]),
        )
        plotly_figure_dict[loss] = fig

    for group in group_losses:
        if group == []:
            continue
        group_means = []
        fig = go.Figure()
        for loss in group:
            means = [
                losses_vs_learning_levels_statistics[loss][level]["mean"]
                for level in levels
            ]
            group_means.append(means)
            if loss == interestingness_criterion:
                if len(interesting_levels) > 0 and len(uninteresting_levels) > 0:
                    interesting_means = [means[level] for level in interesting_levels]
                    uninteresting_means = [means[level] for level in uninteresting_levels]
                    interestingness_ratio = np.mean(interesting_means) / np.mean(
                        uninteresting_means
                    )
            fig.add_trace(go.Bar(x=levels, y=means, name=loss.capitalize() ))
        # add level names
        fig.update_layout(
            title=f"{loss.capitalize()}",
            xaxis_title="Learning level",
            yaxis_title="",
            xaxis=dict(tickvals=levels, ticktext=[level_names[l] for l in levels]),
        )
        plotly_figure_dict[loss] = fig

    recipe._model.train()
    eval_values_dict = {
        "interestingness_ratio": interestingness_ratio,
    }
    if length_generalization is not None:
        eval_values_dict["length_generalization"] = length_generalization
    recipe._model.self_prediction_losses.reset()
    return plotly_figure_dict, eval_values_dict


def visualize_token_values(
    tokens,
    scalar_values,
    new_language=None,
    range=None,
    colormap="inferno",
):
    """
    Generates an HTML string to visualize tokens colored by scalar values.

    This function takes a sequence of tokens and corresponding scalar values (e.g.,
    losses, attention scores) and produces an HTML representation. Each token is
    wrapped in a `<span>` tag with a background color determined by its value,
    scaled to a specified colormap.

    Args:
        tokens (List[str]): The sequence of string tokens to visualize.
        scalar_values (List[float]): A parallel sequence of scalar values that
            determine the color for each token.
        new_language (Optional[List[bool]], optional): A boolean list where `True`
            indicates the start of a new block, forcing a line break. Defaults to None.
        color_range (Optional[Tuple[float, float]], optional): A tuple `(min, max)` to
            use for normalizing the scalar values. If None, the range is automatically
            calculated based on the 95th percentile to handle outliers. Defaults to None.
        colormap (str, optional): The name of the Matplotlib colormap to use.
            Defaults to "inferno".

    Returns:
        str: An HTML string that can be rendered to display the colored tokens.
    """
    if range is None:
        # clip max at the 95th percentile
        sorted_scalar_values = sorted(scalar_values)
        max_value = sorted_scalar_values[int(0.95 * len(sorted_scalar_values))] * 1.5
        min_value = min(sorted_scalar_values)
        range = (min_value, max_value)
        print(f"Range: {min_value} - {max_value}")
    scalar_values = np.array(scalar_values)
    scalar_values = np.clip(scalar_values, range[0], range[1])
    scalar_values = (scalar_values - range[0]) / (range[1] - range[0])

    new_language = (
        np.zeros_like(scalar_values) if new_language is None else new_language
    )

    distance_from_language_break = 0
    distance_from_word_break = 0

    cmap = matplotlib.colormaps[colormap]
    html_text = ""

    for i, (token, value, language_break) in enumerate(
        zip(tokens, scalar_values, new_language)
    ):
        distance_from_word_break += 1
        distance_from_language_break += 1

        color = mcolors.to_hex(cmap(value))
        text_color = "black" if value > 0.7 else "white"
        font_size = 16
        if token == " ":
            token = "__"
            distance_from_word_break = 0

        if language_break and len(html_text) > 0:
            html_text += "<br>"
            distance_from_language_break = 0
            distance_from_word_break = 0
        html_text += (
            f'<span style="background-color:{color}; color:{text_color}; '
            f"padding: 0px 0px; margin: 0px; font-size: {font_size}px; "
            f'font-weight: bold; font-style: regular">{token}</span>'
        )
        if "\n" in token:
            html_text += "<br>"

    return html_text
