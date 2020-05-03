from typing import Tuple, List, Dict, Optional, Union, NewType
import torch
from torch import tensor

import itertools


Bitword = NewType("Bitword", List[int])  # e.g. [0, 1, 1]
Batch = NewType("Batch", Tuple[tensor, tensor])


def pairwise(iterable):
    """
        s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


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


# data loader for distributed environment
from abc import ABC, abstractmethod


class DataFactory(ABC):
    def __init__(self, shard: int, num_shards: int, batch_size: int, sentence_length: int, **kwargs):
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

    @staticmethod
    def _sentence_to_target(sentence: List[Bitword]) -> List[int]:
        return [bitword_to_int(word) for word in sentence]

    def _sentences_to_batches(self, sentences: List[List[Bitword]]) -> List[Batch]:
        targets = tensor([DataFactory._sentence_to_target(sentence) for sentence in sentences])
        sentences = tensor(sentences)

        # split into batch-sized chunks
        targets = torch.split(targets, self.batch_size)
        sentences = torch.split(sentences, self.batch_size)

        return list(zip(sentences, targets))

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

    def filter(self, sentence: tensor, output: tensor) -> Tuple[tensor, tensor]:
        return sentence, output

    def filter_sentence(self, sentence: tensor) -> tensor:
        return sentence
