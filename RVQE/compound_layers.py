from .quantum import *
from .data import int_to_bitword
from .gate_layers import *


import itertools


def powerset(iterable, min_el: int, max_el: int):
    "powerset([1,2,3], 3) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(min_el, max_el)
    )


def index_sets_without(numbers: List[int], exclude: List[int], m: int):
    return powerset([n for n in numbers if n not in exclude], min_el=1, max_el=m + 1)


def _T_gate_list(gates: List[GateLayer]) -> List[GateLayer]:
    return [g.T for g in reversed(gates)]


class CompoundLayer(nn.Module):
    def forward(self, psi: tensor) -> tensor:
        return self.gates.forward(psi)


class BitFlipLayer(CompoundLayer):
    def __init__(self, target_lanes: List[int]):
        super().__init__()
        self.gates = nn.Sequential(*[XLayer(i) for i in target_lanes])


class PostselectManyLayer(CompoundLayer):
    def __init__(self, target_lanes: List[int], on: List[int]):
        super().__init__()
        self.gates = nn.Sequential(*[PostselectLayer(t, w) for t, w in zip(target_lanes, on)])


class UnitaryLayer(CompoundLayer):
    def __init__(self, workspace: List[int]):
        super().__init__()
        self.workspace = workspace
        self.gates = nn.Sequential(
            *[rYLayer(i, initial_θ=0.0) for i in workspace]
        )  # we reverse the order so to have the same lane order as in qiskit

    def extra_repr(self):
        return f"workspace={self.workspace}"


class QuantumNeuronLayer(CompoundLayer):
    def __init__(
        self, workspace: List[int], outlane: int, ancillas: List[int], degree: int = 2, **kwargs
    ):
        """
            workspace from which to take values, write onto outlane; and use ancillas for intermediate computation
            order is implicitly deduced from number of ancillas
            conditions: outlane _can_ be within workspace, but not in ancillas; and workspace and ancillas have to be disjoint
        """
        assert outlane not in ancillas and (
            set(workspace).isdisjoint(ancillas)
        ), "outlane, workspace and ancillas have to be disjoint"
        assert (
            len(workspace) >= 1 and len(ancillas) >= 1
        ), "both workspace and ancillas have to be nonempty"

        super().__init__()

        self.workspace = workspace
        self.ancillas = ancillas
        self.outlane = outlane
        self.order = len(ancillas)
        self.degree = degree

        # precompute parametrized gates as they need to share weights
        self._param_gates = []
        for idcs in index_sets_without(workspace, [outlane], degree):
            self._param_gates.append(crYLayer(idcs, ancillas[0]))
        self._param_gates.append(rYPi4Layer(ancillas[0]))
        self._param_gates.append(rYLayer(ancillas[0], initial_θ=0.))

        # assemble circuit gate layers
        _gates = []
        self._append_gates_recursive(_gates, self.order - 1, dagger=False)
        self.gates = nn.Sequential(*_gates)

    @staticmethod
    def ancillas_for_order(order: int) -> int:
        return order

    def param_gates(self, dagger: bool) -> List[GateLayer]:
        return self._param_gates if not dagger else _T_gate_list(self._param_gates)

    def static_gates(self, idx: int, dagger: bool) -> List[GateLayer]:
        static_lanes = [*self.ancillas, self.outlane]
        static_gates = [cmiYLayer(static_lanes[idx], static_lanes[idx + 1])]

        return static_gates if not dagger else _T_gate_list(static_gates)

    def _append_gates_recursive(self, _gates: List[GateLayer], recidx: int, dagger: bool):
        if recidx == 0:
            _gates += self.param_gates(dagger=False)
            _gates += self.static_gates(0, dagger)
            _gates += self.param_gates(dagger=True)

        else:
            self._append_gates_recursive(_gates, recidx - 1, dagger=False)
            _gates += self.static_gates(recidx, dagger)
            self._append_gates_recursive(_gates, recidx - 1, dagger=True)

        # postselect measurement outcome 0 on corresponding ancilla
        ancilla_to_postselect_on = self.ancillas[recidx]
        _gates.append(PostselectLayer(ancilla_to_postselect_on, on=0))

    def extra_repr(self):
        return f"workspace={self.workspace}, outlane={self.outlane}, ancillas={self.ancillas} (order={self.order}, degree={self.degree})"


def bitword_tensor(width: int) -> tensor:
    """
        returns a tensor in which every row is the binary representation of the row index
    """
    return tensor(
        [[bool(b) for b in int_to_bitword(i, width)] for i in range(2 ** width)], dtype=bool
    )


def subset_index_tensor(width: int, degree: int) -> tensor:
    """
        returns a tensor in which every row is a list of boolean flags that indicate
        whether the row, written in binary, has all those bits set to True that
        appear in the powerset of subsets of length 1...degree
    """
    bwt = bitword_tensor(width)
    setidcs = [list(idcs) for idcs in powerset(range(width), 1, degree + 1)]

    return tensor([[row[idcs].all() for idcs in setidcs] for row in bwt])


class FastQuantumNeuronLayer(nn.Module):
    """
        Mimics the action of a QuantumNeuronLayer, but with a direct implementation
    """

    def __init__(self, workspace: List[int], outlane: int, order: int, degree: int = 2, **kwargs):
        """
            workspace from which to take values, write onto outlane;
            instead of ancilla list we give the order explicitly
            conditions: outlane _can_ be within workspace
        """
        assert order >= 1, "order"
        assert len(workspace) >= 1, "workspace has to be nonempty"

        super().__init__()

        self.workspace = workspace
        self.ancillas = []
        self.outlane = outlane
        self.order = order
        self.degree = degree

        self.sourcelanes = [w for w in workspace if w != outlane]

        # precompute tensor subset indices
        self._ten = subset_index_tensor(len(self.sourcelanes), degree)

        self.φ = nn.Parameter(torch.ones(self._ten.shape[1]))  # weights
        self.θ = nn.Parameter(tensor(0.))  # bias

    @property
    def _sin_cos_op(self) -> Tuple[tensor, tensor]:
        ten = 0.5 * ((self._ten * self.φ).sum(axis=1) + self.θ - pi / 2)

        sin_op = ten.sin() ** (2 ** self.order)
        cos_op = ten.cos() ** (2 ** self.order)

        shape = [2] * len(self.sourcelanes)
        return sin_op.reshape(*shape), cos_op.reshape(*shape)

    def forward(self, psi: tensor) -> tensor:
        sin_op, cos_op = self._sin_cos_op

        # rescale vector with sin and cos prefactors applied
        psi_sin = apply_entrywise(sin_op, psi, self.sourcelanes)
        psi_cos = apply_entrywise(cos_op, psi, self.sourcelanes)

        # move outlane index to end
        psi_cos_t = psi_cos.transpose(self.outlane, 0)
        psi_sin_t = psi_sin.transpose(self.outlane, 0)

        # create rotated variant of vector
        new_psi_0 = psi_cos_t[0] - psi_sin_t[1]
        new_psi_1 = psi_cos_t[1] + psi_sin_t[0]

        # stack and permute last index to its original place
        return normalize(torch.stack([new_psi_0, new_psi_1]).transpose(self.outlane, 0))

    @staticmethod
    def ancillas_for_order(order: int) -> int:
        return 0

    def extra_repr(self):
        return f"workspace={self.workspace}, outlane={self.outlane}, ancillas={self.ancillas} (order={self.order}, degree={self.degree})"
