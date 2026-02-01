# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from pathlib import Path

from typing import Any, Dict, List, Mapping

from torchtune.data import Message

from torchtune.modules.tokenizers import (
    BaseTokenizer,
    ModelTokenizer,
    SentencePieceBaseTokenizer,
    tokenize_messages_no_special_tokens,
)

from torchtune.modules.transforms import Transform


def ascii_tokenizer(
    max_seq_len: int = 2048, add_eos: bool = True, add_bos: bool = True
):
    return ASCIITokenizer(max_seq_len, add_eos, add_bos)


class ASCIITokenizer(BaseTokenizer):
    """
    Simple character-level tokenizer for ASCII text.

    Args:
        max_seq_len (int): Maximum sequence length to truncate tokens to.
        add_eos (bool): Whether to append EOS special token (End of Sentence) to the input.
        add_bos (bool): Whether to prepend BOS special token (Beginning of Sentence) to the input.
    """

    def __init__(
        self, max_seq_len: int = 2048, add_eos: bool = True, add_bos: bool = True
    ):
        self.max_seq_len = max_seq_len
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        self.vocab_size = 128

        self.add_eos = add_eos
        self.add_bos = add_bos

    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        tokens = [ord(c) for c in text]
        if self.add_bos:
            tokens = [self.bos_id] + tokens
        if self.add_eos:
            tokens = tokens + [self.eos_id]

        if self.max_seq_len is not None and len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        return tokens

    def decode(self, token_ids: List[int], **kwargs: Dict[str, Any]) -> str:
        if type(token_ids[0]) == list:
            return [self.decode(t) for t in token_ids]
        else:
            #token_ids = token_ids[1:] if self.add_bos else token_ids
            #token_ids = token_ids[:-1] if self.add_eos else token_ids
            return "".join([chr(i) for i in token_ids])
