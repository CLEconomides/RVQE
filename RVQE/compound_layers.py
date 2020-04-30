from .quantum import *
from .gate_layers import *


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
    def __init__(self, workspace_size: int):
        super().__init__()
        self.gates = nn.Sequential(
            *[rYLayer(i) for i in range(workspace_size)]
        )  # we reverse the order so to have the same lane order as in qiskit

    def extra_repr(self):
        return f"workspace_size={len(self.gates)}"


class QuantumNeuronLayer(CompoundLayer):
    def __init__(self, workspace_size: int, outlane: int, order: int = 2):
        super().__init__()

        self.workspace_size = workspace_size
        self.outlane = outlane
        self.order = order

        self.ancilla_idcs = ancilla_idcs = list(range(workspace_size, workspace_size + order))

        # precompute parametrized gates as they need to share weights
        self._param_gates = []
        for i in range(self.workspace_size):
            if i == self.outlane:
                continue
            self._param_gates.append(crYLayer(i, self.ancilla_idcs[0]))
        self._param_gates.append(rYLayer(self.ancilla_idcs[0]))

        # assemble circuit gate layers
        _gates = []
        self._append_gates_recursive(_gates, self.order - 1, dagger=False)
        self.gates = nn.Sequential(*_gates)

    def param_gates(self, dagger: bool) -> List[GateLayer]:
        return self._param_gates if not dagger else _T_gate_list(self._param_gates)

    def static_gates(self, idx: int, dagger: bool) -> List[GateLayer]:
        static_lanes = [*self.ancilla_idcs, self.outlane]
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
        ancilla_to_postselect_on = self.ancilla_idcs[recidx]
        _gates.append(PostselectLayer(ancilla_to_postselect_on, on=0))

    def extra_repr(self):
        return f"workspace_size={self.workspace_size}, outlane={self.outlane}, order={self.order}"
