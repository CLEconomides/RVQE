from .quantum import apply, tensor, num_operator_qubits, normalize, dot, ket
import torch
from torch import nn

import copy
from math import sqrt

import itertools

_gate_layer_id = 0


class GateLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.dagger = False

        global _gate_layer_id
        _gate_layer_id += 1
        self._id = _gate_layer_id

    @property
    def Ut(self) -> tensor:
        U = self.U
        shape = U.shape
        dim = tensor(U.shape[: len(U.shape) // 2]).prod()  # product of half of the dimensions
        return U.reshape(dim, dim).T.reshape(shape)

    @property
    def T(self):
        new = copy.copy(self)
        new.dagger = not new.dagger
        return new

    def forward(self, psi: tensor, normalize_afterwards: bool = False) -> tensor:
        psi = apply(self.U if not self.dagger else self.Ut, psi, self.lanes)
        return psi if not normalize_afterwards else normalize(psi)

    def extra_repr(self):
        return f"lanes={self.lanes}, id={self._id}{', †' if self.dagger else ''}"

    def to_mat(self):
        n = num_operator_qubits(self.U)
        basis = list(itertools.product("01", repeat=n))

        out = tensor([[dot(ket("".join(x)), self.forward(ket("".join(y)))) for x in basis] for y in basis])

        return out


class XLayer(GateLayer):
    def __init__(self, target_lane: int):
        super().__init__()
        self.lanes = [target_lane]
        self.U = tensor([[0.0, 1.0], [1.0, 0.0]])


class HLayer(GateLayer):
    def __init__(self, target_lane: int):
        super().__init__()
        self.lanes = [target_lane]
        self.U = tensor([[1.0, 1.0], [1.0, -1.0]]) / sqrt(2.0)


class rYLayer(GateLayer):
    def __init__(self, target_lane: int, initial_θ: float = 1.0):
        super().__init__()

        self.lanes = [target_lane]
        self.θ = nn.Parameter(tensor(initial_θ, requires_grad=True))
        self.register_parameter("θ", self.θ)

    @property
    def U(self) -> tensor:
        # note: these matrices are TRANSPOSED! in this notation
        θ = self.θ
        return torch.stack(
            [torch.stack([(0.5 * θ).cos(), (-0.5 * θ).sin()]), torch.stack([(0.5 * θ).sin(), (0.5 * θ).cos()]),]
        )


class crYLayer(GateLayer):
    def __init__(self, control_lane: int, target_lane: int, initial_θ: float = 1.0):
        super().__init__()

        self.lanes = [control_lane, target_lane]
        self.θ = nn.Parameter(tensor(initial_θ, requires_grad=True))
        self.register_parameter("θ", self.θ)

    @property
    def U(self) -> tensor:
        # note: these matrices are TRANSPOSED! in this notation
        θ = self.θ
        return torch.stack(
            [
                torch.stack([tensor([1.0, 0.0]), tensor([0.0, 0.0])]),
                torch.stack([tensor([0.0, 1.0]), tensor([0.0, 0.0])]),
                torch.stack(
                    [tensor([0.0, 0.0]), torch.stack([(0.5 * θ).cos(), (-0.5 * θ).sin()]),]
                ),  # TRANSPOSED again, see comment above
                torch.stack([tensor([0.0, 0.0]), torch.stack([(0.5 * θ).sin(), (0.5 * θ).cos()]),]),
            ]
        ).reshape(2, 2, 2, 2)


class cmiYLayer(GateLayer):
    def __init__(self, control_lane: int, target_lane: int):
        super().__init__()

        self.lanes = [control_lane, target_lane]

        # note: these matrices are TRANSPOSED! in this notation
        self.U = tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0],]
        ).T.reshape(2, 2, 2, 2)


class PostselectLayer(GateLayer):
    def __init__(self, target_lane: int, on: int):
        super().__init__()
        self.on = on
        self.lanes = [target_lane]
        self.U = torch.zeros(4).reshape(2, 2)
        self.U[on, on] = 1.0

    def forward(self, psi: tensor) -> tensor:
        return super().forward(psi, normalize_afterwards=True)

    def extra_repr(self):
        return super().extra_repr() + f", on={self.on}"
