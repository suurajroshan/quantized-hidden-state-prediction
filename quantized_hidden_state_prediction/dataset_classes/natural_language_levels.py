import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data._messages import Message
from torchtune.data._utils import truncate
from torchtune.datasets import SFTDataset, text_completion_dataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

from dataset_classes import messages_pfa_dataset


class TextCompletionDatasetWithMixIn(Dataset):
    """
    A PyTorch Dataset for text completion tasks, with an option to "mix-in" data.

    This dataset loads a primary dataset from a specified source. Additionally,
    it can load text files from a list of "mix-in" directories. During item
    retrieval (__getitem__), with a probability defined by `mix_in_ratio`,
    a sample is randomly drawn from the mixed-in data instead of the primary
    dataset at the given index.

    All samples, whether from the primary or mix-in source, are tokenized
    using the provided tokenizer. The output is a dictionary containing
    'tokens' and 'labels' (a copy of tokens).

    Args:
        tokenizer (ModelTokenizer): The tokenizer to use for encoding text.
        source (str): The source identifier for the primary dataset,
            passed to `datasets.load_dataset`.
        mix_in_sources (List[str]): A list of directory paths containing .txt files
            to be used as mix-in data.
        mix_in_ratio (float, optional): The probability (0.0 to 1.0) of selecting
            a sample from the mix-in data. Defaults to 0.1.
        column (str, optional): The name of the column in the dataset(s)
            containing the text to be tokenized. Defaults to "text".
        add_eos (bool, optional): Whether to add an End-Of-Sequence (EOS) token
            during tokenization. Defaults to True.
        filter_fn (Optional[Callable], optional): An optional function to filter
            samples from the primary dataset. Defaults to None.
        **load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments
            to pass to `datasets.load_dataset` for both primary and mix-in data.
    """
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        mix_in_sources: List[str],
        mix_in_ratio: float = 0.1,
        column: str = "text",
        add_eos: bool = True,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._column = column
        self.add_eos = add_eos

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

        all_mix_in_files = []
        for mix_in_source in mix_in_sources:
            mixed_in_files = [f for f in os.listdir(mix_in_source) if f.endswith(".txt")]
            mixed_in_files = [os.path.join(mix_in_source, f) for f in mixed_in_files]
            all_mix_in_files.extend(mixed_in_files)

        self._mix_in_data = load_dataset("text",
                                         data_files=all_mix_in_files,
                                         sample_by="document",
                                         **load_dataset_kwargs)
        self._mix_in_ratio = mix_in_ratio

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        if torch.rand(1).item() < self._mix_in_ratio:
            sample = self._mix_in_data[torch.randint(0, len(self._mix_in_data), (1,)).item()]
        else:
            sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column]

        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset_with_mix_in(
    tokenizer: ModelTokenizer,
    source: str,
    mix_in_sources: List[str],
    mix_in_ratio: float = 0.1,
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    filter_fn: Optional[Callable] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[TextCompletionDatasetWithMixIn, PackedDataset]:
    """
    Factory function to create a text completion dataset, optionally with mixed-in data and packing.

    This function first instantiates a `TextCompletionDatasetWithMixIn` using the
    provided arguments. If the `packed` argument is True, the resulting dataset
    is then wrapped in a `PackedDataset` to concatenate sequences.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for processing text.
        source (str): Identifier for the primary dataset source.
        mix_in_sources (List[str]): List of directory paths for mix-in data sources.
        mix_in_ratio (float, optional): Probability of using a sample from mix-in
            data. Defaults to 0.1.
        column (str, optional): Column name containing text data. Defaults to "text".
        add_eos (bool, optional): Whether to add an EOS token. Defaults to True.
        packed (bool, optional): If True, wraps the dataset in `PackedDataset`.
            Defaults to False.
        split_across_pack (bool, optional): If packing, whether to split samples
            across pack boundaries. Defaults to True. Used only if `packed` is True.
        split (str, optional): The dataset split to load (e.g., "train", "validation").
            Defaults to "train".
        filter_fn (Optional[Callable], optional): Function to filter samples in the
            primary dataset. Defaults to None.
        **load_dataset_kwargs (Dict[str, Any]): Additional arguments for
            `datasets.load_dataset`.

    Returns:
        Union[TextCompletionDatasetWithMixIn, PackedDataset]: The created dataset,
            either `TextCompletionDatasetWithMixIn` or `PackedDataset` if `packed` is True.

    Raises:
        ValueError: If `packed` is True and `tokenizer.max_seq_len` is not set.
    """
    ds = TextCompletionDatasetWithMixIn(
        tokenizer=tokenizer,
        source=source,
        mix_in_sources=mix_in_sources,
        mix_in_ratio=mix_in_ratio,
        split=split,
        column=column,
        add_eos=add_eos,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack
        )
    return ds


class LanguageLevelsToMessages(Transform):
    """
    A transform that converts a sample from a language levels dataset into a
    structured message format.

    This transform takes an input sample expected to contain 'user' and 'assistant'
    text fields, along with a 'level' identifier. It formats the user and
    assistant text into a list of `Message` objects, where the user's message
    is masked and the assistant's message is not. The original 'level' is
    preserved in the output.

    The output is a dictionary containing the list of messages and the level.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Processes an input sample to create a list of messages.

        Args:
            sample (Mapping[str, Any]): The input data, expected to have keys
                'user' (str), 'assistant' (str), and 'level'.

        Returns:
            Mapping[str, Any]: A dictionary with keys 'messages' (List[Message])
                and 'level'.
        """
        messages = [
            Message(
                role="user",
                content=sample["user"],
                masked=True,
                eot=True,
            ),
            Message(
                role="assistant",
                content=sample["assistant"],
                masked=False,
                eot=True,
            ),
        ]
        return {"messages": messages, "level": sample["level"]}


class LanguageLevelsToTokens(Transform):
    """
    A transform that tokenizes a sample from a language levels dataset using
    a provided tokenizer, while preserving the 'level' information.

    This transform takes an input sample and passes it to the specified
    `ModelTokenizer`. It then adds the 'level' from the input sample to the
    dictionary returned by the tokenizer.

    Args:
        tokenizer (ModelTokenizer): The tokenizer instance to use for converting
            text in the sample to tokens.
    """
    def __init__(self, tokenizer: ModelTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Tokenizes the input sample and appends its 'level'.

        Args:
            sample (Mapping[str, Any]): The input data, expected to be
                compatible with the provided tokenizer and contain a 'level' key.

        Returns:
            Mapping[str, Any]: The dictionary returned by the tokenizer, augmented
                with the original 'level' from the input sample.
        """
        return_dict = self.tokenizer(sample)
        return_dict["level"] = sample["level"]
        return return_dict


def languages_levels_eval_dataset(
    tokenizer: ModelTokenizer,
    location: str
) -> SFTDataset:
    """
    Creates a dataset for evaluating language levels from JSON files.

    This function scans a specified directory (`location`) for all JSON files.
    It then configures an `SFTDataset` to process these files. The processing
    involves two transformation steps:
    1. `LanguageLevelsToMessages`: Converts raw samples into a structured message
       format suitable for language level tasks.
    2. `LanguageLevelsToTokens`: Tokenizes the structured messages using the
       provided `tokenizer`.

    Args:
        tokenizer (ModelTokenizer): The tokenizer instance to use for converting
            text to tokens.
        location (str): The directory path containing the JSON files for the
            evaluation dataset.

    Returns:
        SFTDataset: An instance of `SFTDataset` configured for the language
            levels evaluation task, using data from the specified location.
    """
    model_transform = LanguageLevelsToTokens(tokenizer)
    message_transform = LanguageLevelsToMessages()

    if not os.path.isdir(location):
        raise FileNotFoundError(f"The specified location does not exist or is not a directory: {location}")

    files = [os.path.join(location, f) for f in os.listdir(location) if f.endswith(".json")]

    if not files:
        raise FileNotFoundError(f"No .json files found in the specified location: {location}")

    load_dataset_kwargs = {"data_files": files}
    dataset = SFTDataset(
        source='json', # Assumes SFTDataset can handle 'json' source with data_files
        message_transform=message_transform,
        model_transform=model_transform,
        split='train', # Note: Using 'train' split, which might be conventional for SFTDataset loading
        **load_dataset_kwargs,
    )
    return dataset


class Gsm8kToMessages(Transform):
    """
    Transforms a GSM8K sample into a structured list of messages for chat-based models.

    This transform handles the construction of few-shot prompts, the main question
    (optionally with an added instruction), and the assistant's response based on
    the specified mode ('train', 'test', 'without_answer').

    In 'train' mode, the full answer is included. In 'without_answer' mode, only
    the rationale part of the answer (up to '####') is included. In 'test' mode,
    an empty assistant message is added to prompt the model for a response.

    Args:
        mode (str, optional): Defines how the assistant's answer is processed.
            Must be one of 'train', 'test', or 'without_answer'. Defaults to 'train'.
        few_shot_prompts (Optional[dict], optional): A dictionary or list-like object
            containing few-shot examples, where each example has "question" and "answer" keys.
            Defaults to None.
        num_few_shot_prompts (int, optional): The number of few-shot examples to
            randomly select and prepend to the main question. Defaults to 0.
        prompt_addition (Optional[str], optional): A string to append to the main
            question, often used for chain-of-thought prompting.
            Defaults to "Let's think step by step. At the end, you must write the answer as an integer after '####'.".
    """
    def __init__(self,
                 mode='train',
                 few_shot_prompts: Optional[dict] = None,
                 num_few_shot_prompts: int = 0,
                 prompt_addition: Optional[str] = "Let's think step by step. At the end, you must write the answer as an integer after '####'."):
        super().__init__()
        assert mode in ['train', 'test', 'without_answer']
        self.mode = mode
        self.few_shot_prompt = few_shot_prompts
        self.num_few_shot_prompts = num_few_shot_prompts
        self.prompt_addition = prompt_addition

        if self.few_shot_prompt is not None:
            assert len(self.few_shot_prompt) >= self.num_few_shot_prompts, "Number of few shot prompts should be less than or equal to the number of few shot prompts provided."

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Processes an input GSM8K sample.

        Args:
            sample (Mapping[str, Any]): The input data, expected to have
                'question' and 'answer' keys.

        Returns:
            Mapping[str, Any]: A dictionary containing 'messages' (List[Message]).
                If mode is not 'train', it also includes the original 'answer'.
        """
        messages = []
        # Few shot prompt
        if self.few_shot_prompt is not None:
            # randomly select few shot prompts
            if len(self.few_shot_prompt) == self.num_few_shot_prompts:
                perm = range(self.num_few_shot_prompts)
            else:
                perm = torch.randperm(len(self.few_shot_prompt))[0:self.num_few_shot_prompts]
            for i in perm:
                messages.append(Message(
                    role="user",
                    content=self.few_shot_prompt[i]["question"],
                    masked=True,
                    eot=True,
                ))
                messages.append(Message(
                    role="assistant",
                    content=self.few_shot_prompt[i]["answer"],
                    masked=True,
                    eot=True,
                ))

        # Question
        if self.prompt_addition is not None:
            question = f"{sample['question']} {self.prompt_addition}"
        else:
            question = sample["question"]
        messages.append(Message(
                role="user",
                content=question,
                masked=True,
                eot=True,
            ))

        # Answer / Answer Fragment
        if self.mode == 'train':
            messages.append(Message(
                    role="assistant",
                    content=sample["answer"],
                    masked=False,
                    eot=True,
                ))
        elif self.mode == 'without_answer':
            rationale = sample["answer"]
            rationale = rationale.split('####')[0] + '####'
            messages.append(Message(
                role="assistant",
                content=rationale,
                masked=False,
                eot=False,
            ))
        elif self.mode == 'test':
            messages.append(Message(
                role="assistant",
                content="",
                masked=False,
                eot=False,
            ))

        if not self.mode == 'train':
            return {"messages": messages, "answer": sample["answer"]}
        return {"messages": messages}


class Gsm8kToTokens(Transform):
    """
    Tokenizes a GSM8K sample (typically output from Gsm8kToMessages) and extracts
    the final numerical answer if present.

    This transform applies the provided tokenizer to the input sample. If the sample
    contains an "answer" key (string form), it extracts the numerical part after
    "####", cleans it by removing common non-numeric characters (commas, currency
    symbols, etc.), and converts it to an integer.

    Args:
        tokenizer (ModelTokenizer): The tokenizer instance to use.
    """
    def __init__(self, tokenizer: ModelTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Tokenizes the sample and processes the answer.

        Args:
            sample (Mapping[str, Any]): The input data, expected to be
                compatible with the tokenizer. May contain an "answer" string.

        Returns:
            Mapping[str, Any]: The dictionary returned by the tokenizer, potentially
                augmented with a cleaned, integer "answer" key.
        """
        return_dict = self.tokenizer(sample)
        if "answer" in sample:
            answer = sample["answer"]
            answer = answer.split('####')[-1].strip()
            for remove_char in [',', '$', '%', 'g']:
                answer = answer.replace(remove_char, '')
            return_dict["answer"] = int(answer)
        return return_dict


def gsm_8k_dataset(tokenizer,
                   file,
                   few_shot_prompts=None,
                   num_few_shot_prompts=0,
                   mode='train',
                   prompt_addition="Let's think step by step. At the end, you must write the answer as an integer after '####'."):
    """
    Factory function to create a GSM8K dataset for supervised fine-tuning (SFT).

    This function configures an `SFTDataset` using GSM8K data from a specified file.
    It applies `Gsm8kToMessages` to format the data into a message-based structure
    (optionally including few-shot examples) and then `Gsm8kToTokens` to tokenize
    these messages and process the numerical answer.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for processing text.
        file (str): Path to the JSONL or JSON file containing the GSM8K data.
        few_shot_prompts (Optional[Mapping[int, Dict[str, str]]], optional):
            Few-shot examples to prepend. Each example should be a dictionary
            with "question" and "answer" keys. Defaults to None.
        num_few_shot_prompts (int, optional): Number of few-shot examples to use.
            Defaults to 0.
        mode (str, optional): Operating mode ('train', 'test', 'without_answer')
            for `Gsm8kToMessages`. Defaults to 'train'.
        prompt_addition (Optional[str], optional): Additional text appended to questions.
            Defaults to a chain-of-thought style prompt.

    Returns:
        SFTDataset: An instance of `SFTDataset` configured for GSM8K.
    """
    message_transform = Gsm8kToMessages(few_shot_prompts=few_shot_prompts,
                                        num_few_shot_prompts=num_few_shot_prompts,
                                        mode=mode,
                                        prompt_addition=prompt_addition)
    load_dataset_kwargs = {"data_files": [file]}
    dataset = SFTDataset(
        source='json',
        message_transform=message_transform,
        model_transform=Gsm8kToTokens(tokenizer),
        split='train',
        **load_dataset_kwargs,
    )
    return dataset


class MathToMessages(Transform):
    """
    Transforms a MATH dataset sample into a structured list of messages.

    This transform handles the construction of few-shot prompts, the main problem
    (with an optional instruction), and the assistant's response based on the
    specified mode. It also processes the 'level' and 'type' of the problem,
    converting them into numerical formats.

    Args:
        mode (str, optional): Defines how the assistant's solution is handled.
            Must be one of 'train', 'test', or 'without_answer'. Defaults to 'train'.
        few_shot_prompts (Optional[dict], optional): A dictionary of few-shot
            examples, each with "problem" and "solution" keys. Defaults to None.
        num_few_shot_prompts (int, optional): The number of few-shot examples to
            randomly select. Defaults to 0.
        prompt_addition (Optional[str], optional): A string to append to the
            problem, typically for chain-of-thought prompting. Defaults to a
            prompt asking for a LaTeX solution box.
    """
    def __init__(self,
                 mode='train',
                 few_shot_prompts: Optional[dict] = None,
                 num_few_shot_prompts: int = 0,
                 prompt_addition: Optional[str] = "Let's think step by step. At the end, present your solution in a LaTeX box (i.e., $\\boxed{SOLUTION}$)."):
        super().__init__()
        assert mode in ['train', 'test', 'without_answer']
        self.mode = mode
        self.few_shot_prompt = few_shot_prompts
        self.num_few_shot_prompts = num_few_shot_prompts
        self.prompt_addition = prompt_addition
        self.type_lookup = {
            "Algebra": 0,
            "Counting & Probability": 1,
            "Geometry": 2,
            "Intermediate Algebra": 3,
            "Number Theory": 4,
            "Prealgebra": 5,
            "Precalculus": 6,
        }

        if self.few_shot_prompt is not None:
            assert len(self.few_shot_prompt) >= self.num_few_shot_prompts, "Number of few shot prompts should be less than or equal to the number of few shot prompts provided."

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Processes an input MATH sample to create messages and metadata.

        Args:
            sample (Mapping[str, Any]): The input data, expected to have keys
                'problem', 'solution', 'level', and 'type'.

        Returns:
            Mapping[str, Any]: A dictionary with 'messages' (List[Message]),
                'level' (int), 'type' (int), and optionally the original 'solution' string.
        """
        messages = []
        if self.few_shot_prompt is not None:
            # randomly select few shot prompts
            perm = torch.randperm(len(self.few_shot_prompt))[0:self.num_few_shot_prompts]
            for i in perm:
                messages.append(Message(
                    role="user",
                    content=self.few_shot_prompt[i]["problem"],
                    masked=True,
                    eot=True,
                ))
                messages.append(Message(
                    role="assistant",
                    content=self.few_shot_prompt[i]["solution"],
                    masked=True,
                    eot=True,
                ))
        if self.prompt_addition is not None:
            question = f"{sample['problem']} {self.prompt_addition}"
        else:
            question = sample["problem"]

        messages.append(Message(
                role="user",
                content=question,
                masked=True,
                eot=True,
            ))

        if self.mode == 'train':
            messages.append(Message(
                    role="assistant",
                    content=sample["solution"],
                    masked=False,
                    eot=True,
                ))
        elif self.mode == 'without_answer':
            rationale = sample["solution"]
            rationale = rationale.split('boxed{')[0] + 'boxed{'
            messages.append(Message(
                role="assistant",
                content=rationale,
                masked=False,
                eot=False,
            ))
        elif self.mode == 'test':
            messages.append(Message(
                role="assistant",
                content='',
                masked=False,
                eot=False,
            ))

        level = sample["level"].split('Level ')[-1]
        try:
            level = int(level)
        except ValueError:
            print(f"Error in level: {sample['level']}")
            level = 0
            pass

        return_dict = {"messages": messages,
                       "level": level,
                       "type": self.type_lookup[sample["type"]]}
        if not self.mode == 'train':
            return_dict["solution"] = sample["solution"]
        return return_dict


class MathToTokens(Transform):
    """
    Tokenizes a MATH sample and preserves its metadata.

    This transform applies a tokenizer to a sample (typically the output of
    `MathToMessages`) and ensures that the 'level', 'type', and original
    'solution' string are carried over to the final tokenized output.

    Args:
        tokenizer (ModelTokenizer): The tokenizer instance to use.
    """
    def __init__(self, tokenizer: ModelTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Tokenizes the sample and retains metadata.

        Args:
            sample (Mapping[str, Any]): The input data, expected to be
                compatible with the tokenizer and contain 'level', 'type',
                and optionally 'solution' keys.

        Returns:
            Mapping[str, Any]: The dictionary returned by the tokenizer,
                augmented with the 'level', 'type', and 'solution' from the input.
        """
        return_dict = self.tokenizer(sample)
        return_dict["level"] = sample["level"]
        return_dict["type"] = sample["type"]
        if "solution" in sample:
            solution = sample["solution"]
            # solution = solution.split('boxed{')[-1].split('}$')[0].strip()
            return_dict["solution"] = solution
        return return_dict


def math_dataset(tokenizer,
                 location,
                 few_shot_prompts=None,
                 num_few_shot_prompts=0,
                 mode='train',
                 prompt_addition="Let's think step by step. At the end, present your solution in a LaTeX box (i.e., $\\boxed{SOLUTION}$)."):
    """
    Factory function to create a dataset for the MATH dataset.

    This function recursively finds all JSON files in a given directory, then
    configures and returns an `SFTDataset`. It uses `MathToMessages` and
    `MathToTokens` to transform the raw data into a tokenized format suitable
    for training or evaluation.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for processing text.
        location (str): The root directory to search recursively for `.json` files.
        few_shot_prompts (Optional[dict], optional): Few-shot examples to prepend
            to prompts. Defaults to None.
        num_few_shot_prompts (int, optional): Number of few-shot examples to use.
            Defaults to 0.
        mode (str, optional): Operating mode ('train', 'test', 'without_answer') for
            `MathToMessages`. Defaults to 'train'.
        prompt_addition (Optional[str], optional): Additional text appended to problems.
            Defaults to a chain-of-thought style prompt.

    Returns:
        SFTDataset: An instance of `SFTDataset` configured for the MATH dataset.
    """
    files = []
    for root, _, filenames in os.walk(location):
        for filename in filenames:
            if filename.endswith('.json'):
                files.append(os.path.join(root, filename))

    message_transform = MathToMessages(few_shot_prompts=few_shot_prompts,
                                       num_few_shot_prompts=num_few_shot_prompts,
                                       mode=mode,
                                       prompt_addition=prompt_addition)
    load_dataset_kwargs = {"data_files": files}
    dataset = SFTDataset(
        source='json',
        message_transform=message_transform,
        model_transform=MathToTokens(tokenizer),
        split='train',
        **load_dataset_kwargs,
    )
    return dataset


class CombinedDataset(torch.utils.data.Dataset):
    """
    A dataset that interleaves samples from multiple source datasets.

    This dataset takes a list of PyTorch datasets and combines them by drawing
    samples in a round-robin fashion. For each global index, it determines which
    dataset to sample from and the corresponding index within that dataset.

    Additionally, after retrieving a sample, it ensures that any scalar-like
    metadata (i.e., values that are not lists) are broadcast to match the
    length of the "tokens" sequence in that sample.

    Args:
        datasets (List[torch.utils.data.Dataset]): A list of datasets to combine.
    """
    def __init__(self, datasets: List[torch.utils.data.Dataset]):
        self.datasets = datasets

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        """
        Retrieves an interleaved sample and processes its metadata.

        Args:
            index (int): The global index for the sample to retrieve.

        Returns:
            Dict[str, List[Any]]: The processed sample, where all values are lists
                of the same length as the "tokens" list.
        """
        dataset_index = index % len(self.datasets)
        index_in_dataset = (index // len(self.datasets)) % len(self.datasets[dataset_index])
        sample = self.datasets[dataset_index][index_in_dataset]
        for key, value in sample.items():
            if type(value) != list:
                sample[key] = [value] * len(sample["tokens"])
        return sample


def slim_pajama_dataset(tokenizer):
    """
    Creates a dataset from the SlimPajama-6B source.

    This function is a simple wrapper that calls `text_completion_dataset`
    to load the "train" split of the "DKYoon/SlimPajama-6B" dataset.

    Args:
        tokenizer (ModelTokenizer): The tokenizer to use for the dataset.

    Returns:
        SFTDataset: The loaded text completion dataset.
    """
    dataset = text_completion_dataset(
        tokenizer=tokenizer,
        source="DKYoon/SlimPajama-6B",
        column="text",
        split="train",
    )
    return dataset


def combined_reasoning_and_icl_dataset(tokenizer,
                                       math_location=None,
                                       gsm_location=None,
                                       data_dir=None,):
    """
    Creates a combined dataset for reasoning and in-context learning tasks.

    This function builds and combines four distinct datasets:
    1. MATH dataset for mathematical reasoning.
    2. GSM8K dataset for grade-school math problems.
    3. PFA (Probabilistic Finite Automata) dataset for in-context learning.
    4. SlimPajama for general natural language.

    The final output is a `CombinedDataset` instance that interleaves samples
    from these sources.

    Args:
        tokenizer (ModelTokenizer): The tokenizer to use for all sub-datasets.
        math_location (Optional[str], optional): Path to the MATH dataset's
            training directory. Defaults to a relative path.
        gsm_location (Optional[str], optional): Path to the GSM8K training
            data file. Defaults to a relative path.

    Returns:
        CombinedDataset: A dataset that combines the four specified sources.
    """
    if data_dir is None:
        data_dir = "../data"
    if math_location is None:
        math_location = os.path.join(data_dir, "MATH", "train")
    if gsm_location is None:
        gsm_location = os.path.join(data_dir, "grade-school-math", "grade_school_math", "data", "train.jsonl")
    math_sft_dataset = math_dataset(tokenizer, math_location, mode='train')
    gsm_8k_sft_dataset = gsm_8k_dataset(tokenizer, gsm_location, mode='train')
    pfa_sft_dataset = messages_pfa_dataset(tokenizer)
    pajama = slim_pajama_dataset(tokenizer)
    combined_dataset = CombinedDataset([math_sft_dataset, gsm_8k_sft_dataset, pfa_sft_dataset, pajama])
    return combined_dataset




