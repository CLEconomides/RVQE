from typing import Tuple, List, Dict, Optional, Union, NewType
import torch
from torch import tensor

import itertools


Bitword = NewType("Bitword", List[int])  # e.g. [0, 1, 1]
Batch = NewType("Batch", Tuple[tensor, tensor])


def zip_with_offset(a: tensor, b: tensor, offset: int = 1):
    """
        s -> (a0,b1), (a1,b2), (a2, b3), ...
    """
    return zip(a, b[offset:])


def bitword_to_int(lst: Union[Bitword, tensor]) -> int:
    if not isinstance(lst, list):
        lst = lst.tolist()
    return int("".join(str(n) for n in lst), 2)


def bitword_to_onehot(lst: Union[Bitword, tensor], width: int) -> tensor:
    ret = torch.zeros(2 ** width)
    idx = bitword_to_int(lst)
    ret[idx] = 1.0
    return ret


def int_to_bitword_str(label: int, width: int) -> str:
    return bin(label)[2:].rjust(width, "0")


def int_to_bitword(label: int, width: int) -> Bitword:
    return [int(c) for c in int_to_bitword_str(label, width)]


def bitword_to_str(lst: Union[Bitword, tensor]) -> str:
    return int_to_bitword_str(bitword_to_int(lst), len(lst))


def char_to_bitword(char: str, characters: str, width: int) -> Bitword:
    idx = characters.index(char)
    char_bitword = f"{idx:b}".rjust(width, "0")
    return [int(c) for c in char_bitword[-width:]]


def bitword_to_char(bw: Bitword, characters: str) -> Bitword:
    return characters[bitword_to_int(bw)]


# character error rate
def character_error_rate(sequence: tensor, target: tensor) -> float:
    """
        we assume that sequence and target align 1:1
    """
    assert target.dim() == sequence.dim()
    return 1.0 - (sequence == target).all(dim=-1).to(float).mean()


# target preprocessing helper functions
def targets_for_loss(sentences: tensor):
    """
        batch is B x L x W or just L x W
        B - batch
        L - sentence length
        W - word width
    """
    if sentences.dim() == 2:
        sentences = batch.unsqueeze(0)
    assert sentences.dim() == 3

    return tensor([[bitword_to_int(word) for word in sentence] for sentence in sentences])


def skip_first(targets: tensor) -> tensor:
    """
        we never measure the first one, so skip that
        we assume B x L x W
        B - batch
        L - sentence length
        W - word width
    """
    return targets[:, 1:]


# data loader for distributed environment
from abc import ABC, abstractmethod


class DataFactory(ABC):
    def __init__(
        self, shard: int, num_shards: int, batch_size: int, sentence_length: int, **kwargs
    ):
        self.shard = shard
        self.num_shards = num_shards
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self._index = self.shard  # initial offset dependent on shard

    def next_batch(self) -> Batch:
        # extract batch and advance pointer
        batch = self._batches[self._index]
        self._index += self.num_shards
        self._index %= len(self._batches)

        return batch

    def _sentences_to_batches(
        self, sentences: List[List[Bitword]], targets: List[List[Bitword]]
    ) -> List[Batch]:
        targets = tensor(targets)
        sentences = tensor(sentences)

        # split into batch-sized chunks
        targets = torch.split(targets, self.batch_size)
        sentences = torch.split(sentences, self.batch_size)

        return list(zip(sentences, targets))

    def _sentences_to_batch(
        self, sentences: List[List[Bitword]], targets: List[List[Bitword]]
    ) -> Batch:
        return self._sentences_to_batches(sentences, targets)[0]

    @property
    @abstractmethod
    def _batches(self) -> List[Batch]:
        pass

    @property
    @abstractmethod
    def input_width(self) -> int:
        pass

    @abstractmethod
    def to_human(self, target: tensor, offset: int) -> str:
        """
            translate sentence to a nice human-readable form
            indenting by offset characters
        """
        pass

    def filter(self, sequence: tensor, dim: int) -> tensor:
        return sequence

    def filter_sentence(self, sentence: tensor) -> tensor:
        return sentence

    def ignore_output_at_step(self, index: int, target: Union[Bitword, tensor]) -> bool:
        """
            return True if the output at this step is not expected
            to be a specific target; which means we can postselect it (using OAA)
        """
        return False
