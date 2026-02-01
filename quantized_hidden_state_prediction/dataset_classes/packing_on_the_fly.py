from functools import partial
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from torchtune.data import padded_collate_packed
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from torchtune.datasets._alpaca import alpaca_dataset
from torchtune.models.llama3._tokenizer import Llama3Tokenizer


class PackedOnTheFlyDataset(torch.utils.data.IterableDataset):
    """
    An IterableDataset that packs sequences from an underlying dataset on-the-fly.

    This dataset takes an existing PyTorch Dataset and iterates through it,
    concatenating sequences together until a `max_seq_len` is reached.
    It handles splitting individual samples across multiple packs if necessary
    (controlled by `split_across_pack`) and can process data in a distributed
    manner across multiple workers and ranks. It also supports permuting the
    order of samples from the underlying dataset.

    The output of each iteration is a dictionary (pack) containing 'tokens',
    'labels', 'input_pos', 'seq_lens', and any other sequences present in the
    original dataset, all padded to `max_seq_len`. It also includes metadata like
    '_debug_example_idx' to trace back to original samples.
    """
    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
        permute_indices: bool = False,
        world_size: int = 1,
        rank: int = 0,
        verbose: int = 0,
    ):
        super(PackedOnTheFlyDataset).__init__()
        self.ds = ds
        self.world_size = world_size
        self.rank = rank
        self.verbose = verbose
        if self.world_size > 1:
            start_idx = int(rank * len(self.ds) / world_size)
            end_idx = int((rank + 1) * len(self.ds) / world_size)
            if self.verbose > 0:
                print(
                    f"Rank {rank} trains on indices {start_idx} to {end_idx} (out of {len(self.ds)})"
                )
            self.ds = Subset(self.ds, range(start_idx, end_idx))
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        self.current_idx = 0
        self.current_sample = (
            None  # if we need to split a sample across packs, it will be stored here
        )
        self.current_sample_cutoff = 0  # if we need to split a sample across packs, this will store the last sequence cutoff
        self.end_idx = len(self.ds)
        self.permute_indices = permute_indices
        if self.permute_indices:
            self.dataset_idxs = torch.randperm(len(self.ds))
        else:
            self.dataset_idxs = torch.arange(len(self.ds))

    @staticmethod
    def _worker_init_fn(worker_id, verbose=0):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset

        dataset.current_idx = int(
            worker_info.id * len(dataset.dataset_idxs) / worker_info.num_workers
        )
        dataset.end_idx = int(
            (worker_info.id + 1) * len(dataset.dataset_idxs) / worker_info.num_workers
        )
        if dataset.permute_indices and verbose > 0:
            print(
                f"worker {worker_id} is going through indices {dataset.current_idx} to "
                f"{dataset.end_idx} of permutation {dataset.dataset_idxs[:5]})"
            )
            print(
                f"permutation indices in worker {worker_id}: {dataset.dataset_idxs[dataset.current_idx:dataset.current_idx + 5]}..."
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.end_idx:
            raise StopIteration

        num_tokens = 0
        num_packs = 0
        current_pack = {
            "input_pos": [],
            "_seq_lens": [],
            "_debug_example_idx": [],
        }
        while num_tokens < self.max_seq_len:
            if self.current_idx >= self.end_idx:
                break

            if self.max_packs is not None and num_packs >= self.max_packs:
                break

            if self.current_sample is None:
                self.current_sample = self.ds[
                    self.dataset_idxs[self.current_idx].item()
                ]

            # Dynamically discover sequence and meta-information keys
            keys = self.current_sample.keys()
            is_meta_information = [type(self.current_sample[key]) != list for key in keys]
            meta_information_keys = [key for key, is_meta in zip(keys, is_meta_information) if is_meta]
            sequence_keys = [key for key in keys if key not in meta_information_keys]

            values = [self.current_sample[key] for key in sequence_keys]
            meta_information_values = [self.current_sample[key] for key in meta_information_keys]
            meta_information_keys = ['_' + key for key in meta_information_keys]
            seq_len = len(values[0])
            for key in sequence_keys:
                if key not in current_pack:
                    current_pack[key] = []
            for key in meta_information_keys:
                if key not in current_pack:
                    current_pack[key] = []

            if num_tokens + seq_len <= self.max_seq_len:
                # Add entire (remaining) sample to pack
                for key, value in zip(sequence_keys, values):
                    current_pack[key] += value
                for key, value in zip(meta_information_keys, meta_information_values):
                    current_pack[key] += [value]

                current_pack["input_pos"] += list(range(seq_len))
                current_pack["_seq_lens"] += [seq_len]
                current_pack["_debug_example_idx"] += [
                    self.dataset_idxs[self.current_idx].item()
                ]
                num_tokens += seq_len
                self.current_sample_cutoff = 0
                self.current_idx += 1
                num_packs += 1
                self.current_sample = None
            elif self.split_across_pack or num_tokens == 0:
                # Split sample or add first part if pack is empty
                remaining_sequence_len = self.max_seq_len - num_tokens
                for key, value in zip(sequence_keys, values):
                    current_pack[key] += value[:remaining_sequence_len]
                for key, value in zip(meta_information_keys, meta_information_values):
                    current_pack[key] += [value]

                current_pack["input_pos"] += list(range(remaining_sequence_len))
                current_pack["_seq_lens"] += [remaining_sequence_len]
                current_pack["_debug_example_idx"] += [
                    self.dataset_idxs[self.current_idx].item()
                ]
                num_tokens += remaining_sequence_len
                self.current_sample_cutoff += remaining_sequence_len
                break # Pack is full
            else:
                break # Pack is full

        pack = self._convert_to_tensors(current_pack)
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        pack["seq_lens"] = torch.tensor(pack["_seq_lens"], dtype=torch.long)
        del pack["_seq_lens"]
        return pack

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors."""
        return_dict = {
            key: torch.tensor(value, dtype=torch.long) if key[0] != "_" else value for key, value in pack.items()
        }
        return return_dict

    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        num_padding_tokens = self.max_seq_len - len(pack["tokens"])
        padded_tokens = F.pad(
            pack["tokens"],
            (0, num_padding_tokens),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.max_seq_len - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Add padding tokens as a last seq len to ensure sum is max_seq_len
        padded_seq_lens = pack["_seq_lens"] + [num_padding_tokens] if num_padding_tokens > 0 else pack["_seq_lens"]

        # Pad debug dataset idx
        padded_debug_example_idx = pack["_debug_example_idx"]

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return_dict = {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "_seq_lens": padded_seq_lens,
            "_debug_example_idx": padded_debug_example_idx,
        }

        # account for all additional sequences
        for key in pack.keys():
            if key in ["tokens", "labels", "input_pos", "_seq_lens"]:
                continue

            if key[0] == "_":
                return_dict[key] = pack[key]
                continue

            padded = F.pad(
                pack[key],
                (0, self.max_seq_len - len(pack[key])),
                value=padding_idx,
            )
            return_dict[key] = padded

        return return_dict


if __name__ == "__main__":
    tokenizer = Llama3Tokenizer(
        path="/home/vincent/storage/models/llama3/Meta-Llama-3-8B-Instruct/original/llama3_tokenizer.model",
        max_seq_len=1024,
    )
    ds = alpaca_dataset(tokenizer=tokenizer)
    # ds = Subset(ds, range(1000))
    packed_dataset = PackedOnTheFlyDataset(ds, max_seq_len=1024, permute_indices=True)

    dataloader = DataLoader(
        dataset=packed_dataset,
        batch_size=8,
        collate_fn=partial(
            padded_collate_packed,
        ),
        num_workers=4,
        worker_init_fn=packed_dataset._worker_init_fn,
    )

    for i, batch in enumerate(dataloader):
        print(i)
        print(batch["tokens"].shape)
        print(batch["_debug_example_idx"])
