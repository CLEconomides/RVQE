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


# simple training data


def constant_sentence(length: int, constant: List[int]) -> List[int]:
    return [constant for _ in range(length)]


def alternating_sentence(length: int, constants: List[List[int]]) -> List[int]:
    return [constants[i % len(constants)] for i in range(length)]
