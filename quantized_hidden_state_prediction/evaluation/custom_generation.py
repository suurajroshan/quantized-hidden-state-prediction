from typing import Optional, Tuple, List, Dict
import torch

from torchtune import generation
from torchtune.generation._generation import sample
from torchtune.modules.transformer import TransformerDecoder


ASCII_CODES = [ord(c) for c in 'abcdegfhijklmnopqrstuvwxyz']
ASCII_LOGIT_MASK = torch.ones(128).bool()
ASCII_LOGIT_MASK[ASCII_CODES] = False


def generate_next_token_with_filter(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    logit_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next token given a prompt and returns the corresponding logits,
    with an option to apply a filter (mask) to the logits before sampling.

    Args:
        model (TransformerDecoder): The Transformer model used for generation.
        input_pos (torch.Tensor): A tensor of positional encodings for the prompt,
            with shape `[batch_size x seq_length]`.
        x (torch.Tensor): A tensor of token IDs for the prompt, with shape
            `[batch_size x seq_length]`.
        q (torch.Tensor): A randomly sampled tensor for the softmax sampling trick.
        mask (Optional[torch.Tensor], optional): An attention mask. Defaults to None.
        temperature (float, optional): The value to scale logits by. Defaults to 1.0.
        top_k (Optional[int], optional): The top-k value for sampling. Defaults to None.
        logit_mask (Optional[torch.Tensor], optional): A boolean tensor where `True`
            indices will be set to -inf in the logits, effectively preventing those
            tokens from being sampled. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The generated token IDs, with shape `[batch_size x 1]`.
            - The filtered logits, with shape `[batch_size x seq_length x vocab_size]`.
    """
    # The model produces logits of shape `[batch_size, seq_length, vocab_size]`.
    # We use the logits of the last token in the sequence for the next prediction.
    logits = model(x, input_pos=input_pos, mask=mask)

    if logit_mask is not None:
        # Create a mask to set disabled logits to negative infinity.
        inf_mask = torch.zeros_like(logit_mask, dtype=logits.dtype)
        inf_mask[logit_mask] = -float("inf")
        logits = logits + inf_mask

    # Sample the next token from the last time step's logits.
    next_token = sample(logits[:, -1].clone(), temperature=temperature, top_k=top_k, q=q)
    return next_token, logits


def generate_next_token_only_lowercase(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A specific wrapper around `generate_next_token_with_filter` that constrains
    generation to only lowercase ASCII characters.
    """
    return generate_next_token_with_filter(
        model=model,
        input_pos=input_pos,
        x=x,
        q=q,
        mask=mask,
        temperature=temperature,
        top_k=top_k,
        logit_mask=ASCII_LOGIT_MASK.to(x.device),
    )


def generate(prompt_tokens,
             recipe,
             max_new_tokens=100,
             return_logits=False,
             temperature=0.6,
             top_k=300,
             stop_tokens=[],
             custom_generate_next_token=None):
    """
    Generates token sequences from a model provided within a 'recipe' object.

    This function serves as a high-level wrapper for a generation pipeline. It handles
    batching of multiple prompts, padding them to the same length, calling the core
    generation function, and then decoding and post-processing the results into a
    user-friendly format.

    Args:
        prompt_tokens (List[List[int]]): A list of prompts, where each prompt is a
            list of token IDs.
        recipe (Any): A recipe object that must contain `_model` and `_tokenizer`
            attributes.
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
            Defaults to 100.
        temperature (float, optional): The temperature for sampling. Defaults to 0.6.
        top_k (int, optional): The top-k value for sampling. Defaults to 300.
        stop_tokens (List[int], optional): A list of token IDs that will halt generation
            if produced. Defaults to [].
        custom_generate_next_token (Optional[callable], optional): A custom function for
            next-token generation, useful for applying constraints. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, one for each generated sequence.
            Each dictionary contains:
            - "decoded_tokens" (str): The generated text.
            - "generated_tokens" (torch.Tensor): The raw generated token IDs.
            - "neg_log_probs" (torch.Tensor): The logits of the generated tokens.
    """
    if not type(prompt_tokens[0]) == list:
        prompt_tokens = [prompt_tokens]
    for i, prompt in enumerate(prompt_tokens):
        if prompt[-1] == recipe._tokenizer.eos_id:
            # print("remove eos")
            prompt_tokens[i] = prompt[:-1]
    max_len = max([len(p) for p in prompt_tokens])
    # fill with zeros
    prompt_tokens = [p + [0] * (max_len - len(p)) for p in prompt_tokens]
    prompt = torch.tensor(prompt_tokens, dtype=torch.int, device=recipe._device)

    generated_tokens, generated_logits = generation.generate(
                model=recipe._model,
                prompt=prompt,
                max_generated_tokens=max_new_tokens,
                pad_id=recipe._tokenizer.pad_id,
                temperature=temperature,
                top_k=top_k,
                stop_tokens=stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
    generated_tokens = generated_tokens[:, 1:]

    return_dicts = []
    for i in range(generated_tokens.shape[0]):
        gen_tok = generated_tokens[i]
        gen_log = generated_logits[i]
        mask = gen_tok == recipe._tokenizer.pad_id
        gen_tok = gen_tok[~mask]
        gen_log = gen_log[~mask.to(gen_log.device)]
        decoded_tokens = recipe._tokenizer.decode(gen_tok.tolist())
        #selected_log_probs = gen_log.log_softmax(dim=-1)[range(len(gen_tok)), gen_tok.to(gen_log.device)]
        return_dict = {
            "decoded_tokens": decoded_tokens,
            "generated_tokens": gen_tok,
            "neg_log_probs": gen_log,
        }
        return_dicts.append(return_dict)
    return return_dicts

    # generated_tokens: [bsz x seq_length]
    # generated_logits: [bsz x seq_length x vocab_size]

    selected_neg_log_probs = generated_logits.log_softmax(dim=-1)[
        range(len(generated_tokens[1:])), generated_tokens[1:]
    ]

    decoded_tokens = recipe._tokenizer.decode(generated_tokens.tolist())
    return_dict = {
        "decoded_tokens": decoded_tokens,
        "generated_tokens": generated_tokens,
        "neg_log_probs": selected_neg_log_probs,
    }
    if return_logits:
        return_dict["logits"] = generated_logits
    return return_dict