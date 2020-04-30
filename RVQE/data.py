from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import tensor

import itertools


def pairwise(iterable):
    """
        s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def bin_to_label(lst: Union[List[int], tensor]) -> int:
    if not isinstance(lst, list):
        lst = lst.tolist()

    return int("".join(str(n) for n in lst), 2)


def bin_to_onehot(lst: Union[List[int], tensor], width: int) -> tensor:
    ret = torch.zeros(2 ** width)
    idx = bin_to_label(lst)
    ret[idx] = 1.0
    return ret


def int_to_bin_str(label: int, width: int) -> str:
    return bin(label)[2:].rjust(width, "0")


def int_to_bin(label: int, width: int) -> List[int]:
    return [int(c) for c in int_to_bin_str(label, width)]


def bin_to_str(lst: Union[List[int], tensor]) -> str:
    return int_to_bin_str(bin_to_label(lst), len(lst))


def char_to_bin5(char: str, characters: str) -> List[int]:
    idx = characters.index(char)
    return char_to_bin5.lut[idx]
char_to_bin5.lut = [
    [ int(c) for c in "{0:05b}".format(n) ] for n in range(2**5)
]

# data loader for distributed environment
from abc import ABC, abstractmethod

class DataFactory(ABC):
    def __init__(self, shard: int, num_shards: int, batch_size: int):
        self.shard = shard
        self.num_shards = num_shards
        self.batch_size = batch_size
        self._index = self.shard  # initial offset dependent on shard

    def next_batch(self) -> List[Tuple[tensor]]:
        # extract batch and advance pointer
        batch = self._batches[self._index]
        self._index += self.num_shards
        self._index %= len(self._batches)

        return batch

    def _sentences_to_batches(self, sentences: List[List[int]]) -> List[Tuple[tensor]]:
        targets = tensor([[bin_to_label(word) for word in sentence] for sentence in sentences])
        sentences = tensor(sentences)

        # split into batch-sized chunks
        targets = torch.split(targets, self.batch_size)
        sentences = torch.split(sentences, self.batch_size)

        return list(zip(sentences, targets))

    @property
    @abstractmethod
    def _batches(self) -> List[List[Tuple[tensor]]]:
        pass

    @property
    @abstractmethod
    def input_width(self) -> int:
        pass

    @abstractmethod
    def to_human(self, sentence: tensor, offset: int) -> str:
        """
            translate sentence to a nice human-readable form
            indenting by offset characters
        """
        pass


# simple training data


def constant_sentence(length: int, constant: List[int]) -> List[int]:
    return [constant for _ in range(length)]


def alternating_sentence(length: int, constants: List[List[int]]) -> List[int]:
    return [constants[i % len(constants)] for i in range(length)]


class DataSimpleSentences(DataFactory):
    def __init__(self, sentence_length: int, shard: int, num_shards: int):
        super().__init__(batch_size=max(1, 2 // num_shards), shard=shard, num_shards=num_shards)

        self.sentence_length = sentence_length
        sentences = [
            alternating_sentence(sentence_length, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
            constant_sentence(sentence_length, [1, 0, 0]),
        ]
        self._batches_data = self._sentences_to_batches(sentences)

    @property
    def _batches(self) -> List[Tuple[tensor]]:
        return self._batches_data

    @property
    def input_width(self) -> tensor:
        return 3

    def to_human(self, sentence: tensor, offset: int = 0) -> str:
        return "  "*offset + " ".join([ str(bin_to_label(word)) for word in sentence])



# shakespeare dataset

class DataShakespeare(DataFactory):
    _data = None
    VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! \n"
    DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! ¶"
    assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

    @staticmethod
    def _load_shakespeare():
        import os.path as path
        SHAKESPEARE_PATH = path.join(path.dirname(path.abspath(__file__)), "shakespeare.txt")

        DataShakespeare._data = []

        with open(SHAKESPEARE_PATH, "r") as f:
            leftover = set()

            for i, line in enumerate(f):
                if i < 245 or i > 124440:  # strip license info
                    continue
            
                # cleanup
                line = line.rstrip().lower()
                for c, r in [
                    ("'", ","), (";", ","), ('"', ","), (":", ","),
                    ("1", "one"), ("2", "two"), ("3", "three"), ("4", "four"), ("5", "five"),
                    ("6", "six"), ("7", "seven"), ("8", "eight"), ("9", "nine"), ("0", "zero") ]:
                    line = line.replace(c, r)

                for c in line:
                    if not c in DataShakespeare.VALID_CHARACTERS:
                        c = " "
                    DataShakespeare._data.append(char_to_bin5(c, DataShakespeare.VALID_CHARACTERS))

    

    def __init__(self, sentence_length: int, shard: int, *args, **kwargs):
        super().__init__(shard, *args, **kwargs)

        # local rng
        self.rng = torch.Generator().manual_seed(8742 + shard)

        self.sentence_length = sentence_length
        if self._data == None:
            self._load_shakespeare()

    @property
    def _batches(self) -> List[Tuple[tensor]]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> tensor:
        return 5

    def next_batch(self) -> List[Tuple[tensor]]:
        # extract random batch of sentences
        sentences = []
        while len(sentences) < self.batch_size:
            idx_start = torch.randint(len(self._data) - self.sentence_length, (1,), generator=self.rng).item()
            sentences.append(self._data[idx_start : idx_start + self.sentence_length])

        # turn into batch
        return self._sentences_to_batches(sentences)[0]
        
    def to_human(self, sentence: tensor, offset: int = 0) -> str:
        return " "*offset + "".join([ DataShakespeare.DISPLAY_CHARACTERS[bin_to_label(c)] for c in sentence ])
