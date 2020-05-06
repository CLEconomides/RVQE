from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import tensor
from math import pi, sqrt

import functools


def ket0(num_qubits: int) -> tensor:
    psi = torch.zeros(2 ** num_qubits)
    psi[0] = 1
    return psi.reshape(shape=[2] * num_qubits)


def ket1(num_qubits: int) -> tensor:
    psi = torch.zeros(2 ** num_qubits)
    psi[-1] = 1
    return psi.reshape(shape=[2] * num_qubits)


def ket(descr: str) -> tensor:
    out = None
    for s in descr:
        if s == "0":
            psi = tensor([1.0, 0.0])
        elif s == "1":
            psi = tensor([0.0, 1.0])
        elif s == "+":
            psi = normalize(tensor([1.0, 1.0]))
        elif s == "-":
            psi = normalize(tensor([1.0, -1.0]))
        else:
            assert False, "description has to be one of 0, 1, + or -"

        if out is None:
            out = psi
        else:
            out = torch.ger(out, psi).view(-1)

    return out.reshape(shape=[2] * len(descr))


def normalize(psi: tensor) -> tensor:
    return psi / psi.norm(p=2)


def num_state_qubits(psi: tensor) -> int:
    return len(psi.shape)


def num_operator_qubits(op: tensor) -> int:
    assert len(op.shape) % 2 == 0, "operator does not have same input and output indices"
    return len(op.shape) // 2


_EINSUM_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def squish_idcs_up(idcs: str) -> str:
    sorted_idcs = sorted(idcs)
    return "".join([_EINSUM_ALPHABET[-i - 1] for i in [len(idcs) - 1 - sorted_idcs.index(c) for c in idcs]])


@functools.lru_cache(maxsize=10 ** 6)
def einsum_indices(m: int, n: int, target_lanes: Tuple[int]) -> Tuple[str, str, str]:
    assert len(target_lanes) == m, "number of target and operator indices don't match"
    assert torch.all(tensor(target_lanes) < n), "target lanes not present in state"

    idcs_op = squish_idcs_up("".join(_EINSUM_ALPHABET[-l - 1] for l in target_lanes)) + "".join(
        _EINSUM_ALPHABET[r] for r in target_lanes
    )
    idcs_target = _EINSUM_ALPHABET[:n]

    assert len(idcs_op) + len(idcs_target) < len(_EINSUM_ALPHABET), "too few indices for torch's einsum"

    idcs_result = ""
    idcs_op_lut = dict(zip(idcs_op[m:], idcs_op[:m]))  # lookup table from operator's right to operator's left indices
    for c in idcs_target:
        if c in idcs_op_lut:
            idcs_result += idcs_op_lut[c]
        else:
            idcs_result += c

    return (idcs_op, idcs_target, idcs_result)


def probabilities(psi: tensor, measured_lanes: Optional[List[int]] = None, verbose: bool = False):
    if measured_lanes is None:
        measured_lanes = range(len(psi.shape))
    n = num_state_qubits(psi)

    idcs_kept = "".join(_EINSUM_ALPHABET[i] for i in measured_lanes)
    idcs_einsum = f"{ _EINSUM_ALPHABET[:n] },{ _EINSUM_ALPHABET[:n] }->{idcs_kept}"
    verbose and print(idcs_einsum)
    return torch.einsum(idcs_einsum, psi, psi).reshape(-1)


def apply(op: tensor, psi: tensor, target_lanes: List[int], verbose: bool = False) -> tensor:
    n = num_state_qubits(psi)
    m = num_operator_qubits(op)

    idcs_op, idcs_target, idcs_result = einsum_indices(m, n, tuple(target_lanes))
    idcs_einsum = f"{idcs_op},{idcs_target}->{idcs_result}"
    verbose and print(idcs_einsum)

    return torch.einsum(idcs_einsum, op, psi)


def dot(a: tensor, b: tensor) -> tensor:
    idcs_einsum = f"{_EINSUM_ALPHABET[:len(a.shape)]},{_EINSUM_ALPHABET[:len(b.shape)]}->"
    return torch.einsum(idcs_einsum, a, b)


def ctrlMat(op: tensor, num_control_lanes: int) -> tensor:
    if num_control_lanes == 0:
        return op
    n = num_operator_qubits(op)
    A = torch.eye(2 ** n)
    AB = torch.zeros(2 ** n, 2 ** n)
    BA = torch.zeros(2 ** n, 2 ** n)
    return ctrlMat(
        torch.cat([torch.cat([A, AB], dim=0), torch.cat([BA, op.reshape(2 ** n, -1)], dim=0)], dim=1).reshape(
            *[2] * (2 * (n + 1))
        ),
        num_control_lanes - 1,
    )
