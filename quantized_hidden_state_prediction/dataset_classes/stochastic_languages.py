from typing import Any, Mapping, Optional

import numpy as np
import torch.utils.data

from torchtune.data._messages import Message
from torchtune.data._utils import truncate
from torchtune.modules.tokenizers import BaseTokenizer, ModelTokenizer
from torchtune.modules.transforms import Transform

from models.tokenizer import ascii_tokenizer


class PFADataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that procedurally generates sequences from random Probabilistic
    Finite Automata (PFAs) for in-context language learning tasks.

    Each item from the dataset consists of a long sequence created by concatenating
    multiple shorter character sequences. These shorter sequences are drawn from
    several randomly generated "languages," where each language is defined by a unique PFA.
    The dataset keeps track of the start of each new language and includes the
    definitions of the PFAs themselves for evaluation.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for encoding the generated character sequences.
        num_states_min (int, optional): Minimum number of states in a PFA. Defaults to 4.
        num_states_max (int, optional): Maximum number of states in a PFA. Defaults to 12.
        alphabet_size_min (int, optional): Minimum number of unique characters (vocabulary size)
            for a PFA. Defaults to 4.
        alphabet_size_max (int, optional): Maximum number of unique characters for a PFA.
            Defaults to 18.
        seq_len_min (int, optional): Minimum length of a single character sequence generated
            from a PFA. Defaults to 1.
        seq_len_max (int, optional): Maximum length of a single character sequence.
            Defaults to 50.
        edges_per_state_min (int, optional): Minimum number of outgoing edges from any state
            in a PFA. Defaults to 1.
        edges_per_state_max (int, optional): Maximum number of outgoing edges. Defaults to 4.
        sequences_per_language_min (int, optional): Minimum number of example sequences to
            generate from a single PFA. Defaults to 10.
        sequences_per_language_max (int, optional): Maximum number of example sequences.
            Defaults to 20.
        max_sample_length (int, optional): The target character length for the concatenated
            sample returned by __getitem__. Defaults to 2048.
    """
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        num_states_min: int = 4,
        num_states_max: int = 12,
        alphabet_size_min: int = 4,
        alphabet_size_max: int = 18,
        seq_len_min: int = 1,
        seq_len_max: int = 50,
        edges_per_state_min: int = 1,
        edges_per_state_max: int = 4,
        sequences_per_language_min: int = 10,
        sequences_per_language_max: int = 20,
        max_sample_length: int = 2048,
    ):
        super(PFADataset).__init__()
        self.tokenizer = tokenizer
        self.num_states_min = num_states_min
        self.num_states_max = num_states_max
        self.alphabet_size_min = alphabet_size_min
        self.alphabet_size_max = alphabet_size_max
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max
        self.edges_per_state_min = edges_per_state_min
        self.edges_per_state_max = edges_per_state_max
        self.sequences_per_language_min = sequences_per_language_min
        self.sequences_per_language_max = sequences_per_language_max
        self.max_sample_length = max_sample_length

        self.tokens = torch.arange(self.alphabet_size_max)
        self.letters = "abcdefghijklmnopqrstuvwxyz"

        assert self.alphabet_size_max <= len(
            self.letters
        ), "Alphabet size is too large for the number of letters"

    def __getitem__(self, item):
        """
        Generates and returns a single training sample composed of multiple languages.

        Args:
            item (int): The index of the item to retrieve (ignored, as data is
                generated on the fly).

        Returns:
            dict: A dictionary containing the tokenized sample and metadata:
                - "tokens" (List[int]): The tokenized sequence.
                - "labels" (List[int]): A copy of the tokens for language modeling.
                - "new_language" (List[bool]): A boolean list indicating the
                  start of a new PFA's sequences.
                - "transition_probs" (List[List[List[float]]]): The transition probability
                  matrices of the generated PFAs.
                - "transition_symbols" (List[List[List[int]]]): The transition symbol
                  matrices of the generated PFAs.
        """
        length = 0
        sample = []
        new_language_flags = []
        last_transition_probs, last_transition_symbols = [], []
        while length < self.max_sample_length:
            num_sequences_for_this_language = torch.randint(
                self.sequences_per_language_min, self.sequences_per_language_max, (1,)
            ).item()
            transition_probs, transition_symbols = self.generate_language()
            last_transition_probs.append(transition_probs.tolist())
            last_transition_symbols.append(transition_symbols.tolist())
            language_sequences = []
            for _ in range(num_sequences_for_this_language):
                sequence, _ = self.generate_sequence(
                    transition_probs, transition_symbols
                )
                language_sequences.append(sequence)
            new_language = " ".join(language_sequences) + " "
            sample.append(new_language)
            length += len(new_language)
            new_language = torch.zeros(len(new_language), dtype=torch.bool)
            new_language[0] = True
            new_language_flags.append(new_language)

        sample = "".join(sample[:-1])[:-1]
        new_language_flags = torch.cat(new_language_flags[:-1]).tolist()[:-1]

        tokens = self.tokenizer.encode(sample)
        if self.tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self.tokenizer.max_seq_len - 1)
        labels = tokens.copy()

        new_language_flags = [False] + new_language_flags + [False]

        return {"tokens": tokens,
                "labels": labels,
                "new_language": new_language_flags,
                "transition_probs": last_transition_probs,
                "transition_symbols": last_transition_symbols}

    def generate_language(self):
        """Creates a single random Probabilistic Finite Automaton (PFA)."""
        vocab_size = torch.randint(
            self.alphabet_size_min, self.alphabet_size_max, (1,)
        ).item()
        vocabulary = torch.arange(self.alphabet_size_max)[
            torch.randperm(self.alphabet_size_max)[:vocab_size]
        ]
        num_states = torch.randint(
            self.num_states_min, self.num_states_max, (1,)
        ).item()
        states = torch.arange(num_states + 1)  # state 0 is the start state

        # construct the probabilistic finite automaton
        # - for every state, sample between 1 and 4 edges to other states
        # - every edge is assigned with a word from the vocabulary (chosen uniformly at random)
        transition_probs = torch.zeros((num_states + 1, num_states + 1))
        transition_symbols = (
            torch.zeros((num_states + 1, num_states + 1), dtype=torch.int64) - 1
        )
        for state in states:
            num_outgoing_edges = torch.randint(
                self.edges_per_state_min, self.edges_per_state_max, (1,)
            ).item()
            outgoing_edges = states[torch.randperm(num_states)[:num_outgoing_edges] + 1]
            outgoing_symbols = vocabulary[
                torch.randperm(vocab_size)[:num_outgoing_edges]
            ]
            transition_probs[state, outgoing_edges] = 1 / num_outgoing_edges
            transition_symbols[state, outgoing_edges] = outgoing_symbols

        return transition_probs, transition_symbols

    def generate_sequence(self, transition_probs, transition_symbols):
        """Generates a single character sequence from a given PFA."""
        sequence_length = torch.randint(self.seq_len_min, self.seq_len_max, (1,))
        sequence = []
        sequence_log_prob = 0
        current_state = 0
        for _ in range(sequence_length):
            next_state = torch.multinomial(transition_probs[current_state], 1).item()
            next_symbol = transition_symbols[current_state, next_state].item()
            sequence.append(next_symbol)
            sequence_log_prob += torch.log(transition_probs[current_state, next_state])
            current_state = next_state

        string_sequence = "".join([self.letters[s] for s in sequence])
        return string_sequence, sequence_log_prob

    def __len__(self):
        return 10_000_000


def pfa_dataset(
    tokenizer: BaseTokenizer,
    num_states_min: int = 4,
    num_states_max: int = 12,
    alphabet_size_min: int = 4,
    alphabet_size_max: int = 18,
    seq_len_min: int = 1,
    seq_len_max: int = 50,
    edges_per_state_min: int = 1,
    edges_per_state_max: int = 4,
    sequences_per_language_min: int = 10,
    sequences_per_language_max: int = 20,
    max_sample_length: int = 2048,
):
    """
    Factory function to create a PFADataset for in-context language learning.

    This function initializes and returns a `PFADataset`, which procedurally
    generates data based on random Probabilistic Finite Automata (PFAs).
    All arguments are passed directly to the `PFADataset` constructor.

    Args:
        tokenizer (BaseTokenizer): Tokenizer for encoding the generated character sequences.
        num_states_min (int, optional): Minimum number of states in a PFA. Defaults to 4.
        num_states_max (int, optional): Maximum number of states in a PFA. Defaults to 12.
        alphabet_size_min (int, optional): Minimum vocabulary size for a PFA. Defaults to 4.
        alphabet_size_max (int, optional): Maximum vocabulary size for a PFA. Defaults to 18.
        seq_len_min (int, optional): Minimum length of a single sequence from a PFA. Defaults to 1.
        seq_len_max (int, optional): Maximum length of a single sequence. Defaults to 50.
        edges_per_state_min (int, optional): Minimum outgoing edges from any state. Defaults to 1.
        edges_per_state_max (int, optional): Maximum outgoing edges from any state. Defaults to 4.
        sequences_per_language_min (int, optional): Minimum example sequences per PFA. Defaults to 10.
        sequences_per_language_max (int, optional): Maximum example sequences per PFA. Defaults to 20.
        max_sample_length (int, optional): Target character length for a concatenated sample. Defaults to 2048.

    Returns:
        PFADataset: An instance of the configured PFADataset.
    """
    ds = PFADataset(
        tokenizer=tokenizer,
        num_states_min=num_states_min,
        num_states_max=num_states_max,
        alphabet_size_min=alphabet_size_min,
        alphabet_size_max=alphabet_size_max,
        seq_len_min=seq_len_min,
        seq_len_max=seq_len_max,
        edges_per_state_min=edges_per_state_min,
        edges_per_state_max=edges_per_state_max,
        sequences_per_language_min=sequences_per_language_min,
        sequences_per_language_max=sequences_per_language_max,
        max_sample_length=max_sample_length,
    )
    return ds


class LearningLevelsPFADataset(PFADataset):
    """
    Generates sequences from Probabilistic Finite Automata (PFAs) across multiple,
    distinct levels of learning complexity.

    This dataset is designed for studying in-context learning by creating tasks that
    range from simple memorization to complex, on-the-fly inference. It extends
    `PFADataset` by defining several "learning levels," each corresponding to a
    different type of sequence generation task. When an item is requested, it
    constructs a sample by randomly mixing blocks of sequences from these different levels.

    The defined learning levels are:
    - **Level 0 (Memorized Sequences):** Retrieves pre-generated, fixed sequences from a
      fixed set of automata.
    - **Level 1 (Memorized Programs):** Generates new sequences on-the-fly, but from a
      fixed, pre-generated set of automata.
    - **Level 2 (Fixed Structure, New Vocab):** Uses pre-generated automaton structures
      but swaps their character vocabularies at generation time.
    - **Level 3 (In-Context Learning):** Generates sequences from entirely new, random
      automata on-the-fly.
    - **Level 4 (Random):** Generates sequences of completely random characters, with no
      underlying structure.
    - **Level 5 (Copying):** Generates random sequences and then repeats them, creating a
      copying task.

    Args:
        tokenizer (ModelTokenizer, optional): Tokenizer for encoding the generated sequences.
            If None, raw character strings are returned. Defaults to None.
        num_states_min (int, optional): Min states in a generated PFA. Defaults to 4.
        num_states_max (int, optional): Max states in a generated PFA. Defaults to 12.
        alphabet_size_min (int, optional): Min vocabulary size for a PFA. Defaults to 4.
        alphabet_size_max (int, optional): Max vocabulary size for a PFA. Defaults to 18.
        seq_len_min (int, optional): Min length of a single generated sequence. Defaults to 1.
        seq_len_max (int, optional): Max length of a single generated sequence. Defaults to 50.
        edges_per_state_min (int, optional): Min outgoing edges from a PFA state. Defaults to 1.
        edges_per_state_max (int, optional): Max outgoing edges from a PFA state. Defaults to 4.
        sequences_per_language_min (int, optional): Min sequences per language block. Defaults to 10.
        sequences_per_language_max (int, optional): Max sequences per language block. Defaults to 20.
        max_sample_length (int, optional): Target character length for a full sample. Defaults to 2048.
        num_fixed_automata (int, optional): Number of fixed PFAs to pre-generate for levels
            0, 1, and 2. Defaults to 10.
        num_fixed_sequences (int, optional): Number of fixed sequences to generate per PFA
            for level 0. Defaults to 10.
        fixed_seed (int, optional): A seed for the random number generator to ensure the
            fixed automata and sequences are the same across runs. Defaults to 123.
        shuffle_random_sequences (bool, optional): For level 5, if True, all copied sequences
            are shuffled together. If False, each sequence is repeated immediately. Defaults to True.
        word_perturbation_rate (float, optional): Probability of applying token-level
            perturbations to a generated sequence. Defaults to 0.5.
        token_perturbation_rate (float, optional): If a sequence is perturbed, this is the
            probability that any given token within it will be replaced by a random one. Defaults to 0.2.
        included_learning_levels (tuple, optional): A tuple of integer learning levels (0-5)
            to sample from when generating data. Defaults to (0, 1, 2, 3, 4, 5).
        max_num_languages_per_sample (Optional[int], optional): The maximum number of different
            language/level blocks to include in a single sample. Defaults to None (infinite).
        constrained_sequences (int, optional): If > 0, for this many initial sequences in a PFA
            block, a more predictable (constrained) version of the PFA is used. Defaults to 0.
        edge_constrain_ratio (float, optional): The ratio of edges to temporarily disable to
            create a constrained PFA. Defaults to 0.2.
    """
    def __init__(
        self,
        tokenizer: ModelTokenizer = None,
        num_states_min: int = 4,
        num_states_max: int = 12,
        alphabet_size_min: int = 4,
        alphabet_size_max: int = 18,
        seq_len_min: int = 1,
        seq_len_max: int = 50,
        edges_per_state_min: int = 1,
        edges_per_state_max: int = 4,
        sequences_per_language_min: int = 10,
        sequences_per_language_max: int = 20,
        max_sample_length: int = 2048,
        num_fixed_automata: int = 10,
        num_fixed_sequences: int = 10,
        fixed_seed: int = 123,
        shuffle_random_sequences: bool = True,
        word_perturbation_rate: float = 0.5,
        token_perturbation_rate: float = 0.2,
        included_learning_levels=(0, 1, 2, 3, 4, 5),
        max_num_languages_per_sample: Optional[int] = None,
        constrained_sequences: int = 0,
        edge_constrain_ratio: float = 0.2,
    ):
        super().__init__(
            tokenizer=tokenizer,
            num_states_min=num_states_min,
            num_states_max=num_states_max,
            alphabet_size_min=alphabet_size_min,
            alphabet_size_max=alphabet_size_max,
            seq_len_min=seq_len_min,
            seq_len_max=seq_len_max,
            edges_per_state_min=edges_per_state_min,
            edges_per_state_max=edges_per_state_max,
            sequences_per_language_min=sequences_per_language_min,
            sequences_per_language_max=sequences_per_language_max,
            max_sample_length=max_sample_length,
        )

        self.word_perturbation_rate = word_perturbation_rate
        self.token_perturbation_rate = token_perturbation_rate
        self.included_learning_levels = included_learning_levels
        if max_num_languages_per_sample is None:
            self.max_num_languages_per_sample = float("inf")
        else:
            self.max_num_languages_per_sample = max_num_languages_per_sample
        self.constrained_sequences = constrained_sequences
        self.edge_constrain_ratio = edge_constrain_ratio
        self.shuffle_random_sequences = shuffle_random_sequences

        # There are three levels:
        # - Level 0: fixed sequences from fixed automata
        # - Level 1: generated sequences from fixed automata
        # - Level 2: generated sequences from random automata with fixed structure (different vocabulary)
        # - Level 3: generates sequences from random automata
        # - Level 4: generates completely random sequences
        # - Level 5: generates random sequences and repeats them (in shuffled order)

        # The next part needs a fixed random seed
        seed_state = torch.get_rng_state()
        torch.manual_seed(fixed_seed)

        # Level 0: Generate languages and save fixed sequences from them
        self.level0_languages = [
            self.generate_language() for _ in range(num_fixed_automata)
        ]
        self.level0_sequences = []
        for transition_probs, transition_symbols in self.level0_languages:
            s = []
            for _ in range(num_fixed_sequences):
                sequence, _ = self.generate_sequence(
                    transition_probs, transition_symbols
                )
                s.append(sequence)
            self.level0_sequences.append(s)

        # Level 1: Generate languages
        self.level1_languages = [
            self.generate_language() for _ in range(num_fixed_automata)
        ]

        # Level 2: Generate langauges (to be modified when the sequence is generated)
        self.level2_languages = [
            self.generate_language() for _ in range(num_fixed_automata)
        ]

        # Levels 3, 4 & 5: Everything is generated on-the-fly, nothing to do here

        # Return to initial random state
        torch.set_rng_state(seed_state)

    def __getitem__(self, item):
        """
        Generates and returns a single training sample composed of sequences from
        various learning levels.

        Args:
            item (int): The index of the item (ignored, as data is generated on-the-fly).

        Returns:
            dict: A dictionary containing the tokenized sample and extensive metadata,
                with all values being lists of the same length as the token sequence:
                - "tokens" (List[int]): The tokenized sequence.
                - "labels" (List[int]): A copy of the tokens for language modeling.
                - "new_language" (List[bool]): A boolean flag, True at the start of a new
                  language/level block.
                - "learning_level" (List[int]): An integer (0-5) indicating the complexity
                  level of each token.
                - "num_states" (List[int]): The number of states in the PFA that generated
                  the token (0 for non-PFA levels).
                - "num_edges" (List[int]): The number of edges in the PFA.
                - "vocab_size" (List[int]): The vocabulary size of the PFA.
                - "perturbation" (List[bool]): A flag indicating if a token was randomly
                  perturbed from its original value.
                - "_transition_probs" (List): The transition matrices of the PFAs used.
                - "_transition_symbols" (List): The symbol matrices of the PFAs used.
        """
        length = 0
        sample = []
        new_language_flags = []
        learning_level = []
        perturbation = []
        num_states = []
        num_edges = []
        vocab_size = []
        last_transition_probs, last_transition_symbols = [], []
        num_language_in_sample = 0
        while (length < self.max_sample_length) and (num_language_in_sample < self.max_num_languages_per_sample + 1):
            num_sequences_for_this_language = torch.randint(
                self.sequences_per_language_min, self.sequences_per_language_max, (1,)
            ).item()

            # Choose learning level
            current_level = self.included_learning_levels[
                torch.randint(len(self.included_learning_levels), (1,)).item()
            ]

            if current_level == 0:
                all_sequences_from_language = self.level0_sequences[
                    torch.randint(len(self.level0_sequences), (1,)).item()
                ]
                seq_idxs = torch.randint(
                    len(all_sequences_from_language), (num_sequences_for_this_language,)
                )
                language_sequences = [
                    all_sequences_from_language[idx] for idx in seq_idxs
                ]
                l_num_states, l_num_edges, l_effective_vocab_size = (0, 0, 0)
            elif current_level == 4:
                # create random sequences
                language_sequences = []
                for _ in range(num_sequences_for_this_language):
                    sequence_length = torch.randint(
                        self.seq_len_min, self.seq_len_max, (1,)
                    ).item()
                    sequence = torch.randint(
                        self.alphabet_size_max, (sequence_length,)
                    ).tolist()
                    language_sequences.append(
                        "".join([self.letters[s] for s in sequence])
                    )
                l_num_states, l_num_edges, l_effective_vocab_size = (0, 0, 0)
            elif current_level == 5:
                # create half the number of random sequences, and repeat them
                half_number_of_sequences = num_sequences_for_this_language // 2
                original_sequences = []
                for _ in range(half_number_of_sequences):
                    sequence_length = torch.randint(
                        self.seq_len_min, self.seq_len_max, (1,)
                    ).item()
                    sequence = torch.randint(
                        self.alphabet_size_max, (sequence_length,)
                    ).tolist()
                    original_sequences.append(
                        "".join([self.letters[s] for s in sequence])
                    )
                if self.shuffle_random_sequences:
                    language_sequences = original_sequences * 2
                    language_sequences = np.random.permutation(language_sequences)
                else:
                    # repeat each sequence directly after the original
                    language_sequences = []
                    for seq in original_sequences:
                        language_sequences.append(seq)
                        language_sequences.append(seq)
                l_num_states, l_num_edges, l_effective_vocab_size = (0, 0, 0)
            else:
                if current_level == 1:
                    transition_probs, transition_symbols = self.level1_languages[
                        torch.randint(len(self.level1_languages), (1,)).item()
                    ]
                elif current_level == 2:
                    transition_probs, transition_symbols = self.level2_languages[
                        torch.randint(len(self.level2_languages), (1,)).item()
                    ]
                    # change the transition symbols
                    l2_vocab_size = torch.randint(
                        self.alphabet_size_min, self.alphabet_size_max, (1,)
                    ).item()
                    vocabulary = torch.arange(self.alphabet_size_max)[
                        torch.randperm(self.alphabet_size_max)[:l2_vocab_size]
                    ]
                    for state in range(transition_symbols.shape[0]):
                        edge_exists = transition_symbols[state] >= 0
                        num_outgoing_edges = edge_exists.sum()
                        outgoing_symbols = vocabulary[
                            torch.randperm(l2_vocab_size)[:num_outgoing_edges]
                        ]
                        transition_symbols[state][edge_exists] = outgoing_symbols
                elif current_level == 3:
                    transition_probs, transition_symbols = self.generate_language()

                language_sequences = []
                # compute metadata for the language
                l_num_states = transition_probs.shape[0]
                l_num_edges = (transition_symbols >= 0).sum().item()
                l_effective_vocab_size = len(
                    torch.unique(transition_symbols[transition_symbols >= 0])
                )

                if self.constrained_sequences == 0:
                    for _ in range(num_sequences_for_this_language):
                        sequence, _ = self.generate_sequence(
                            transition_probs, transition_symbols
                        )
                        language_sequences.append(sequence)
                else:
                    constrained_transition_probs = transition_probs.clone()
                    constraint_mask = torch.rand_like(constrained_transition_probs) < self.edge_constrain_ratio
                    constrained_transition_probs[constraint_mask] = 0.
                    # If there is a row with all zeros, replace it with the unconstrained row
                    for i in range(constrained_transition_probs.shape[0]):
                        if constrained_transition_probs[i].sum() == 0:
                            constrained_transition_probs[i] = transition_probs[i]
                    # normalize the rows
                    constrained_transition_probs /= constrained_transition_probs.sum(dim=1, keepdim=True)
                    for i in range(num_sequences_for_this_language):
                        if i < self.constrained_sequences:
                            sequence, _ = self.generate_sequence(
                                constrained_transition_probs, transition_symbols
                            )
                        else:
                            sequence, _ = self.generate_sequence(
                                transition_probs, transition_symbols
                            )
                        language_sequences.append(sequence)

                last_transition_probs.append(transition_probs.tolist())
                last_transition_symbols.append(transition_symbols.tolist())


            # Perturbation
            for i, seq in enumerate(language_sequences):
                if torch.rand(1) > self.word_perturbation_rate:
                    perturbation.append(torch.zeros(len(seq) + 1, dtype=bool))
                    continue
                seq = list(seq)
                (perturb_idxs,) = torch.where(
                    torch.rand(len(seq)) < self.token_perturbation_rate
                )
                perturb_flag = torch.zeros(len(seq) + 1, dtype=bool)
                for pi in perturb_idxs:
                    seq[pi] = self.letters[
                        torch.randint(self.alphabet_size_max, (1,)).item()
                    ]
                    perturb_flag[pi] = True
                language_sequences[i] = "".join(seq)
                perturbation.append(perturb_flag)

            language_sequences_string = " ".join(language_sequences) + " "
            sample.append(language_sequences_string)
            length += len(language_sequences_string)

            new_language = torch.zeros(len(language_sequences_string), dtype=torch.bool)
            new_language[0] = True
            new_language_flags.append(new_language)

            learning_level.append(torch.ones_like(new_language) * current_level)
            num_states.append(torch.ones_like(new_language) * l_num_states)
            num_edges.append(torch.ones_like(new_language) * l_num_edges)
            vocab_size.append(torch.ones_like(new_language) * l_effective_vocab_size)

            num_language_in_sample += 1

        sample = "".join(sample[:-1])
        # remove last language
        new_language_flags = new_language_flags[:-1]
        learning_level = learning_level[:-1]
        num_states = num_states[:-1]
        num_edges = num_edges[:-1]
        vocab_size = vocab_size[:-1]
        last_transition_probs = last_transition_probs[:-1]
        last_transition_symbols = last_transition_symbols[:-1]

        if self.tokenizer is None:
            tokens = sample
            labels = sample
        else:
            # Tokenize (BOS and EOS tokens will be added)
            tokens = self.tokenizer.encode(sample)
            if self.tokenizer.max_seq_len is not None:
                tokens = truncate(tokens, self.tokenizer.max_seq_len - 1)
            # Labels are identical to tokens, the shift for autoregressive modelling happens in the _loss_step() function
            # in the recipe
            labels = tokens.copy()

        # add BOS and EOS information
        new_language_flags = (
            [False] + torch.cat(new_language_flags).tolist() + [False]
        )
        learning_level = [-1] + torch.cat(learning_level).tolist() + [-1]
        num_states = [-1] + torch.cat(num_states).tolist() + [-1]
        num_edges = [-1] + torch.cat(num_edges).tolist() + [-1]
        vocab_size = [-1] + torch.cat(vocab_size).tolist() + [-1]
        perturbation = (
            [False]
            + torch.cat(perturbation)[: len(learning_level) - 2].tolist()
            + [False]
        )

        return_dict = {
            "tokens": tokens,
            "labels": labels,
            "new_language": new_language_flags,
            "learning_level": learning_level,
            "num_states": num_states,
            "num_edges": num_edges,
            "vocab_size": vocab_size,
            "perturbation": perturbation,
            "_transition_probs": last_transition_probs,
            "_transition_symbols": last_transition_symbols,
        }

        return return_dict


def learning_levels_pfa_dataset(
    tokenizer: BaseTokenizer,
    num_states_min: int = 4,
    num_states_max: int = 12,
    alphabet_size_min: int = 4,
    alphabet_size_max: int = 18,
    seq_len_min: int = 1,
    seq_len_max: int = 50,
    edges_per_state_min: int = 1,
    edges_per_state_max: int = 4,
    sequences_per_language_min: int = 10,
    sequences_per_language_max: int = 20,
    max_sample_length: int = 2048,
    num_fixed_automata: int = 10,
    num_fixed_sequences: int = 10,
    fixed_seed: int = 123,
    word_perturbation_rate: float = 0.5,
    token_perturbation_rate: float = 0.2,
    included_learning_levels=(0, 1, 2, 3, 4, 5),
    max_num_languages_per_sample: Optional[int] = None,
    constrained_sequences: int = 0,
    edge_constrain_ratio: float = 0.2,
    shuffle_random_sequences: bool = True,
):
    """
    Factory function to create a `LearningLevelsPFADataset`.

    This function initializes and returns an instance of `LearningLevelsPFADataset`,
    which generates complex sequences based on Probabilistic Finite Automata (PFAs)
    across multiple, distinct levels of learning complexity. All arguments are
    passed directly to the `LearningLevelsPFADataset` constructor.

    Args:
        tokenizer (BaseTokenizer): Tokenizer for encoding generated sequences.
        num_states_min (int, optional): Min states in a PFA. Defaults to 4.
        num_states_max (int, optional): Max states in a PFA. Defaults to 12.
        alphabet_size_min (int, optional): Min vocabulary size for a PFA. Defaults to 4.
        alphabet_size_max (int, optional): Max vocabulary size for a PFA. Defaults to 18.
        seq_len_min (int, optional): Min length of a single sequence. Defaults to 1.
        seq_len_max (int, optional): Max length of a single sequence. Defaults to 50.
        edges_per_state_min (int, optional): Min outgoing edges from a state. Defaults to 1.
        edges_per_state_max (int, optional): Max outgoing edges from a state. Defaults to 4.
        sequences_per_language_min (int, optional): Min sequences per language block. Defaults to 10.
        sequences_per_language_max (int, optional): Max sequences per language block. Defaults to 20.
        max_sample_length (int, optional): Target character length for a full sample. Defaults to 2048.
        num_fixed_automata (int, optional): Number of fixed PFAs for levels 0, 1, and 2. Defaults to 10.
        num_fixed_sequences (int, optional): Number of fixed sequences per PFA for level 0. Defaults to 10.
        fixed_seed (int, optional): Seed for generating fixed automata and sequences. Defaults to 123.
        word_perturbation_rate (float, optional): Probability of perturbing a sequence. Defaults to 0.5.
        token_perturbation_rate (float, optional): Probability of perturbing a token in a perturbed sequence. Defaults to 0.2.
        included_learning_levels (Tuple[int, ...], optional): Which learning levels (0-5) to sample from.
            Defaults to (0, 1, 2, 3, 4, 5).
        max_num_languages_per_sample (Optional[int], optional): Max number of language blocks per sample.
            Defaults to None (unlimited).
        constrained_sequences (int, optional): Number of initial sequences in a block to generate from
            a more predictable PFA. Defaults to 0.
        edge_constrain_ratio (float, optional): Ratio of PFA edges to disable for constrained generation.
            Defaults to 0.2.
        shuffle_random_sequences (bool, optional): For the copying task (level 5), whether to shuffle
            the repeated sequences. Defaults to True.

    Returns:
        LearningLevelsPFADataset: An instance of the configured dataset.
    """
    ds = LearningLevelsPFADataset(
        tokenizer=tokenizer,
        num_states_min=num_states_min,
        num_states_max=num_states_max,
        alphabet_size_min=alphabet_size_min,
        alphabet_size_max=alphabet_size_max,
        seq_len_min=seq_len_min,
        seq_len_max=seq_len_max,
        edges_per_state_min=edges_per_state_min,
        edges_per_state_max=edges_per_state_max,
        sequences_per_language_min=sequences_per_language_min,
        sequences_per_language_max=sequences_per_language_max,
        max_sample_length=max_sample_length,
        num_fixed_automata=num_fixed_automata,
        num_fixed_sequences=num_fixed_sequences,
        fixed_seed=fixed_seed,
        word_perturbation_rate=word_perturbation_rate,
        token_perturbation_rate=token_perturbation_rate,
        included_learning_levels=included_learning_levels,
        max_num_languages_per_sample=max_num_languages_per_sample,
        constrained_sequences=constrained_sequences,
        edge_constrain_ratio=edge_constrain_ratio,
        shuffle_random_sequences=shuffle_random_sequences,
    )
    return ds


class FormalLanguageToMessages(Transform):
    """
    Transforms a raw PFA sample into a message-based format for an LLM.

    This transform takes a sample containing a character string, converts the
    string into a custom integer representation, and wraps it in a standard
    user/assistant prompt structure. The user prompt is a fixed instruction,
    and the assistant prompt is left empty for the model to complete.

    Args:
        prompt (str, optional): The fixed user prompt to prepend to the sample.
            Defaults to a prompt about generating examples from a PFA.
    """
    def __init__(self,
                 prompt = "Generate examples from a formal language defined by a probabilistic finite automaton. As tokens, use any of the lowercase letters of the alphabet.",):
        super().__init__()
        self.prompt = prompt
        self.char_lookup = {c: i + 64 for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
        self.char_lookup[" "] = 220

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Processes an input sample containing a PFA-generated string.

        Args:
            sample (Mapping[str, Any]): The input data, expected to have a "tokens"
                key containing the character string.

        Returns:
            Mapping[str, Any]: A dictionary with "messages" (List[Message]) and
                "tokenized_sample" (a list of custom integer codes).
        """
        messages = []

        sample = sample["tokens"]
        tokenized_sample = [self.char_lookup[c] for c in sample]

        messages.append(Message(
                role="user",
                content=self.prompt,
                masked=True,
                eot=True,
            ))

        messages.append(Message(
                role="assistant",
                content='',
                masked=False,
                eot=True,
            ))
        return {"messages": messages, "tokenized_sample": tokenized_sample}


class LearningLevelsMessagesDataset(LearningLevelsPFADataset):
    """
    A dataset that formats PFA-generated data into prompts for a large language model.

    This class builds upon `LearningLevelsPFADataset` to create a specialized dataset
    for in-context learning of formal languages. It is hardcoded to only generate
    data from "Level 3" (new, random PFAs) and ensures each sample comes from a single PFA.

    The key operation is in `__getitem__`, where it takes a PFA-generated sequence,
    wraps it in a user/assistant prompt, and then splices a custom integer
    representation of the sequence into the tokenized prompt. This creates a single
    input sequence for the model, combining the instructional prompt with the raw
    formal language data. All associated metadata is padded to match the final
    token length.

    Args:
        tokenizer (ModelTokenizer, optional): The LLM tokenizer for processing the
            user/assistant prompts. Defaults to None.
    """
    def __init__(self,
                 tokenizer: ModelTokenizer = None,
                 num_states_min: int = 4,
                 num_states_max: int = 12,
                 alphabet_size_min: int = 4,
                 alphabet_size_max: int = 18,
                 seq_len_min: int = 1,
                 seq_len_max: int = 50,
                 edges_per_state_min: int = 1,
                 edges_per_state_max: int = 4,
                 sequences_per_language_min: int = 10,
                 sequences_per_language_max: int = 20,
                 max_sample_length: int = 2048):
        super().__init__(
            tokenizer=None,
            num_states_min=num_states_min,
            num_states_max=num_states_max,
            alphabet_size_min=alphabet_size_min,
            alphabet_size_max=alphabet_size_max,
            seq_len_min=seq_len_min,
            seq_len_max=seq_len_max,
            edges_per_state_min=edges_per_state_min,
            edges_per_state_max=edges_per_state_max,
            sequences_per_language_min=sequences_per_language_min,
            sequences_per_language_max=sequences_per_language_max,
            max_sample_length=max_sample_length,
            word_perturbation_rate=0.,
            token_perturbation_rate=0,
            included_learning_levels=(3,),
            max_num_languages_per_sample=1,)

        self.llm_tokenizer = tokenizer
        self.message_transform = FormalLanguageToMessages()

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        messages = self.message_transform(sample)
        tokenized = self.llm_tokenizer(messages)
        tokens = tokenized['tokens']
        final_tokens = tokens[-2:]
        prefix_tokens = tokens[:-3]
        tokens = prefix_tokens + tokenized['tokenized_sample'] + final_tokens
        sample['tokens'] = tokens
        sample['mask'] = tokenized['mask'] + [False] * (len(sample['tokens']) - len(tokenized['mask']))
        sample['labels'] = tokens

        num_tokens = len(tokens)
        # the length of the additional info should be the same as the length of the tokens
        for k, v in sample.items():
            if k not in ['tokens', 'labels', 'mask']:
                num_missing_values = max(0, num_tokens - len(v) - 1)
                sample[k] = [v[0]] * num_missing_values + v

        return sample


def messages_pfa_dataset(
    tokenizer: BaseTokenizer,
    num_states_min: int = 4,
    num_states_max: int = 12,
    alphabet_size_min: int = 4,
    alphabet_size_max: int = 18,
    seq_len_min: int = 1,
    seq_len_max: int = 50,
    edges_per_state_min: int = 1,
    edges_per_state_max: int = 4,
    sequences_per_language_min: int = 10,
    sequences_per_language_max: int = 20,
    max_sample_length: int = 2048,
):
    """
    Factory function to create a `LearningLevelsMessagesDataset`.

    This function initializes a dataset specifically for prompting LLMs with
    in-context learning tasks based on Probabilistic Finite Automata (PFAs).
    All arguments are passed directly to the `LearningLevelsMessagesDataset` constructor.

    Args:
        tokenizer (BaseTokenizer): The LLM tokenizer for processing prompts.
        // Other args control the generation of the underlying PFAs.

    Returns:
        LearningLevelsMessagesDataset: An instance of the configured dataset.
    """
    ds = LearningLevelsMessagesDataset(
        tokenizer=tokenizer,
        num_states_min=num_states_min,
        num_states_max=num_states_max,
        alphabet_size_min=alphabet_size_min,
        alphabet_size_max=alphabet_size_max,
        seq_len_min=seq_len_min,
        seq_len_max=seq_len_max,
        edges_per_state_min=edges_per_state_min,
        edges_per_state_max=edges_per_state_max,
        sequences_per_language_min=sequences_per_language_min,
        sequences_per_language_max=sequences_per_language_max,
        max_sample_length=max_sample_length,
    )
    return ds


class RandomStringsDataset(torch.utils.data.Dataset):
    """
    A Dataset that generates sequences by concatenating perturbed random strings.

    This class first creates a fixed pool of random character strings. When an
    item is requested, it repeatedly samples from this pool, applies a random
    amount of perturbation (changing some characters), and concatenates the
    results into a single long sequence up to a maximum length.

    This dataset is useful for creating a baseline task where there is no
    learnable structure, corresponding to the "Random Examples" task for evaluating
    in-context computation.

    Args:
        tokenizer (ModelTokenizer): Tokenizer for encoding the final character sequence.
        alphabet_size_min (int, optional): Minimum size of the character alphabet
            for generating a base string. Defaults to 4.
        alphabet_size_max (int, optional): Maximum size of the character alphabet.
            Defaults to 18.
        seq_len_min (int, optional): Minimum length of a single base string.
            Defaults to 1.
        seq_len_max (int, optional): Maximum length of a single base string.
            Defaults to 50.
        num_strings (int, optional): The number of unique base strings to pre-generate
            and store in the pool. Defaults to 100.
        max_perturbation_rate (float, optional): The maximum fraction of characters
            in a string that can be randomly changed (perturbed) before use.
            The actual rate is chosen uniformly from [0, max_perturbation_rate].
            Defaults to 0.2.
        max_sample_length (int, optional): The target character length for the final
            concatenated sample returned by __getitem__. Defaults to 2048.
    """
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        alphabet_size_min: int = 4,
        alphabet_size_max: int = 18,
        seq_len_min: int = 1,
        seq_len_max: int = 50,
        num_strings: int = 100,
        max_perturbation_rate: float = 0.2,
        max_sample_length: int = 2048,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.alphabet_size_min = alphabet_size_min
        self.alphabet_size_max = alphabet_size_max
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max
        self.max_sample_length = max_sample_length
        self.max_perturbation_rate = max_perturbation_rate

        self.strings = [self.generate_string() for _ in range(num_strings)]

        self.tokens = torch.arange(self.alphabet_size_max)
        self.letters = "abcdefghijklmnopqrstuvwxyz"

    def generate_string(self):
        """Generates a single random string as a tensor of character indices."""
        vocab_size = torch.randint(
            self.alphabet_size_min, self.alphabet_size_max, (1,)
        ).item()
        vocabulary = torch.arange(self.alphabet_size_max)[
            torch.randperm(self.alphabet_size_max)[:vocab_size]
        ]
        sequence_length = torch.randint(self.seq_len_min, self.seq_len_max, (1,))
        sequence = vocabulary[torch.randint(0, vocab_size, (sequence_length,))]
        return sequence

    def __getitem__(self, item):
        """
        Generates and returns a single sample of concatenated random strings.

        Args:
            item (int): The index of the item (ignored, as data is generated on-the-fly).

        Returns:
            dict: A dictionary containing the tokenized sample and metadata:
                - "tokens" (List[int]): The tokenized sequence.
                - "labels" (List[int]): A copy of the tokens.
                - "new_language" (List[bool]): A boolean list marking the start
                  of each new perturbed string in the sequence.
        """
        length = 0
        sample = []
        new_language_flags = []
        while length < self.max_sample_length:
            idx = torch.randint(0, len(self.strings), (1,)).item()
            string = self.strings[idx]
            perturbation_rate = torch.rand(1).item() * self.max_perturbation_rate
            num_perturbations = int(len(string) * perturbation_rate)
            perturbation_indices = torch.randperm(len(string))[:num_perturbations]
            perturbation_values = torch.randint(
                0, self.alphabet_size_max, (num_perturbations,)
            )
            string[perturbation_indices] = perturbation_values
            string = "".join([self.letters[s] for s in string]) + " "
            sample.append(string)
            length += len(string)
            new_language = torch.zeros(len(string), dtype=torch.bool)
            new_language[0] = True
            new_language_flags.append(new_language)

        sample = "".join(sample[:-1])[:-1]
        new_language_flags = torch.cat(new_language_flags[:-1]).tolist()[:-1]

        tokens = self.tokenizer.encode(sample)
        if self.tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self.tokenizer.max_seq_len - 1)
        labels = tokens.copy()

        new_language_flags = [False] + new_language_flags + [False]

        return {"tokens": tokens, "labels": labels, "new_language": new_language_flags}

    def __len__(self):
        return 10_000_000


class MixedPFARandomDataset(PFADataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        num_states_min: int = 4,
        num_states_max: int = 12,
        alphabet_size_min: int = 4,
        alphabet_size_max: int = 18,
        seq_len_min: int = 1,
        seq_len_max: int = 50,
        edges_per_state_min: int = 1,
        edges_per_state_max: int = 4,
        sequences_per_language_min: int = 10,
        sequences_per_language_max: int = 20,
        max_sample_length: int = 2048,
        num_strings: int = 100,
        max_perturbation_rate: float = 0.2,
    ):
        """
        A Dataset that generates samples by mixing PFA-based and random sequences.

        This class creates training samples by interleaving blocks of sequences from two
        different sources, chosen with a 50/50 probability for each block:
        1.  **PFA Sequences**: Generated on-the-fly from a new, random Probabilistic
            Finite Automaton (PFA), representing a structured, learnable task.
        2.  **Random Sequences**: Drawn from a pre-generated pool of random strings,
            with a random amount of perturbation applied.

        This setup is useful for training a model on a mixture of structured and
        unstructured data. The dataset provides a "pfa" flag to identify the
        source of each token.

        Args:
            tokenizer (ModelTokenizer): Tokenizer for encoding the final character sequence.
            // PFA-related args (num_states_min, etc.) are inherited from PFADataset.
            num_strings (int, optional): The number of unique base strings to pre-generate
                for the random sequence pool. Defaults to 100.
            max_perturbation_rate (float, optional): The maximum fraction of characters
                in a random string that can be perturbed. Defaults to 0.2.
        """
        super().__init__(
            tokenizer=tokenizer,
            num_states_min=num_states_min,
            num_states_max=num_states_max,
            alphabet_size_min=alphabet_size_min,
            alphabet_size_max=alphabet_size_max,
            seq_len_min=seq_len_min,
            seq_len_max=seq_len_max,
            edges_per_state_min=edges_per_state_min,
            edges_per_state_max=edges_per_state_max,
            sequences_per_language_min=sequences_per_language_min,
            sequences_per_language_max=sequences_per_language_max,
            max_sample_length=max_sample_length,
        )

        # generate random strings with fixed seed
        seed_state = torch.get_rng_state()
        torch.manual_seed(123)
        self.strings = [self.generate_string() for _ in range(num_strings)]
        torch.set_rng_state(seed_state)

        self.max_perturbation_rate = max_perturbation_rate

    def generate_string(self):
        vocab_size = torch.randint(
            self.alphabet_size_min, self.alphabet_size_max, (1,)
        ).item()
        vocabulary = torch.arange(self.alphabet_size_max)[
            torch.randperm(self.alphabet_size_max)[:vocab_size]
        ]
        sequence_length = torch.randint(self.seq_len_min, self.seq_len_max, (1,))
        sequence = vocabulary[torch.randint(0, vocab_size, (sequence_length,))]
        return sequence

    def __getitem__(self, item):
        """
        Generates a single sample by mixing PFA and random sequence blocks.

        Args:
            item (int): The index of the item (ignored).

        Returns:
            dict: A dictionary containing the tokenized sample and metadata:
                - "tokens" (List[int]): The final tokenized sequence.
                - "labels" (List[int]): A copy of the tokens.
                - "new_language" (List[bool]): A flag, True at the start of each new block.
                - "pfa" (List[bool]): A flag, True if the token belongs to a PFA-generated
                  sequence, and False if it belongs to a random one.
        """
        length = 0
        sample = []
        new_language_flags = []
        pfa_flags = []
        while length < self.max_sample_length:
            num_sequences_for_this_language = torch.randint(
                self.sequences_per_language_min, self.sequences_per_language_max, (1,)
            ).item()
            if torch.rand(1).item() < 0.5:
                transition_probs, transition_symbols = self.generate_language()
                language_sequences = []
                for _ in range(num_sequences_for_this_language):
                    sequence, _ = self.generate_sequence(
                        transition_probs, transition_symbols
                    )
                    language_sequences.append(sequence)
                new_language = " ".join(language_sequences) + " "
                new_pfa_flags = torch.ones(len(new_language), dtype=torch.bool)
            else:
                sequences = []
                for _ in range(num_sequences_for_this_language):
                    idx = torch.randint(0, len(self.strings), (1,)).item()
                    string = self.strings[idx]
                    perturbation_rate = (
                        torch.rand(1).item() * self.max_perturbation_rate
                    )
                    num_perturbations = int(len(string) * perturbation_rate)
                    perturbation_indices = torch.randperm(len(string))[
                        :num_perturbations
                    ]
                    perturbation_values = torch.randint(
                        0, self.alphabet_size_max, (num_perturbations,)
                    )
                    string[perturbation_indices] = perturbation_values
                    string = "".join([self.letters[s] for s in string])
                    sequences.append(string)
                new_language = " ".join(sequences) + " "
                new_pfa_flags = torch.zeros(len(new_language), dtype=torch.bool)

            sample.append(new_language)
            length += len(new_language)
            new_language = torch.zeros(len(new_language), dtype=torch.bool)
            new_language[0] = True
            new_language_flags.append(new_language)
            pfa_flags.append(new_pfa_flags)

        sample = "".join(sample[:-1])[:-1]
        new_language_flags = torch.cat(new_language_flags[:-1]).tolist()[:-1]
        pfa_flags = torch.cat(pfa_flags[:-1]).tolist()[:-1]

        tokens = self.tokenizer.encode(sample)
        if self.tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self.tokenizer.max_seq_len - 1)
        labels = tokens.copy()

        new_language_flags = [False] + new_language_flags + [False]
        pfa_flags = [False] + pfa_flags + [False]

        return {
            "tokens": tokens,
            "labels": labels,
            "new_language": new_language_flags,
            "pfa": pfa_flags,
        }


def mixed_pfa_random_dataset(
    tokenizer: BaseTokenizer,
    num_states_min: int = 4,
    num_states_max: int = 12,
    alphabet_size_min: int = 4,
    alphabet_size_max: int = 18,
    seq_len_min: int = 1,
    seq_len_max: int = 50,
    edges_per_state_min: int = 1,
    edges_per_state_max: int = 4,
    sequences_per_language_min: int = 10,
    sequences_per_language_max: int = 11,
    max_sample_length: int = 2048,
    num_strings: int = 100,
    max_perturbation_rate: float = 0.0,
) -> MixedPFARandomDataset:
    """
    Factory function to create a `MixedPFARandomDataset`.

    This function initializes a dataset that generates samples by mixing structured,
    PFA-based sequences with unstructured, random sequences. All arguments are
    passed directly to the `MixedPFARandomDataset` constructor.

    Args:
        tokenizer (BaseTokenizer): Tokenizer for encoding the final character sequences.
        // Other args control the generation of the PFA and random string components.

    Returns:
        MixedPFARandomDataset: An instance of the configured dataset.
    """
    ds = MixedPFARandomDataset(
        tokenizer=tokenizer,
        num_states_min=num_states_min,
        num_states_max=num_states_max,
        alphabet_size_min=alphabet_size_min,
        alphabet_size_max=alphabet_size_max,
        seq_len_min=seq_len_min,
        seq_len_max=seq_len_max,
        edges_per_state_min=edges_per_state_min,
        edges_per_state_max=edges_per_state_max,
        sequences_per_language_min=sequences_per_language_min,
        sequences_per_language_max=sequences_per_language_max,
        max_sample_length=max_sample_length,
        num_strings=num_strings,
        max_perturbation_rate=max_perturbation_rate,
    )
    return ds


if __name__ == "__main__":
    tokenizer = ascii_tokenizer()
    dataset = LearningLevelsPFADataset(tokenizer)
    for i in range(10):
        a = dataset[i]["tokens"]
        decoded_tokens = dataset.tokenizer.decode(a)
        pass

    pass
