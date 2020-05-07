from .quantum import *
from .gate_layers import *


import itertools


def powerset(iterable, min_el: int, max_el: int):
    "powerset([1,2,3], 3) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(min_el, max_el))


def index_sets_without(numbers: List[int], exclude: List[int], m: int):
    return powerset(set(numbers) - set(exclude), min_el=1, max_el=m + 1)


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
    def __init__(self, workspace: List[int], outlane: int, ancillas: List[int], degree: int = 2):
        """
            workspace from which to take values, write onto outlane; and use ancillas for intermediate computation
            conditions: outlane _can_ be within workspace, but not in ancillas; and workspace and ancillas have to be disjoint
        """
        assert outlane not in ancillas and (
            set(workspace).isdisjoint(ancillas)
        ), "outlane, workspace and ancillas have to be disjoint"
        assert len(workspace) >= 1 and len(ancillas) >= 1, "both workspace and ancillas have to be nonempty"

        super().__init__()

        self.workspace = workspace
        self.ancillas = ancillas
        self.outlane = outlane
        self.order = len(ancillas)

        # precompute parametrized gates as they need to share weights
        self._param_gates = []
        for idcs in index_sets_without(workspace, [outlane], degree):
            self._param_gates.append(crYLayer(idcs, ancillas[0]))
        self._param_gates.append(rYLayer(ancillas[0], initial_θ=pi / 4))

        # assemble circuit gate layers
        _gates = []
        self._append_gates_recursive(_gates, self.order - 1, dagger=False)
        self.gates = nn.Sequential(*_gates)

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
        return f"workspace={self.workspace}, outlane={self.outlane}, ancillas={self.ancillas} (order={self.order})"
