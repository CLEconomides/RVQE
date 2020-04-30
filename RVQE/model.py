from typing import Tuple, List, Dict, Optional, Union

from .compound_layers import (
    UnitaryLayer,
    QuantumNeuronLayer,
    BitFlipLayer,
    PostselectManyLayer,
)
from .quantum import tensor, ket0, probabilities
from .data import pairwise

import torch
from torch import nn


class RVQECell(nn.Module):
    def __init__(self, workspace_size: int, stages: int, order: int = 2):
        super().__init__()

        self.workspace_size = workspace_size
        self.order = order

        self.gates = nn.Sequential(
            *[
                nn.Sequential(
                    UnitaryLayer(workspace_size),
                    *[
                        QuantumNeuronLayer(workspace_size, outlane, order)
                        for outlane in range(workspace_size)
                    ]
                )
                for _ in range(stages)
            ]
        )

    @property
    def num_qubits(self) -> int:
        return self.workspace_size + self.order

    def forward(self, psi: tensor, input: List[int]) -> Tuple[tensor, tensor]:
        # we assume psi has its input lanes reset to 0
        input_lanes = range(len(input))
        psi = BitFlipLayer([i for i in input_lanes if input[i] == 1]).forward(psi)
        psi = self.gates.forward(psi)

        return probabilities(psi, input_lanes), psi


class RVQE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cell = RVQECell(*args, **kwargs)

    def forward(
        self, inputs: tensor, postselect_measurement: bool = True
    ) -> Tuple[tensor, list]:
        # batched call; return stacked result
        if inputs.dim() == 3:
            batch_measured_seq = []
            batch_probs = []
            for inpt in inputs:
                probs, measured_seq = self.forward(inpt, postselect_measurement)
                batch_probs.append(probs)
                batch_measured_seq.append(measured_seq)
            return torch.stack(batch_probs), torch.stack(batch_measured_seq)

        # normal call
        assert (
            inputs.dim() == 2
        ), "inputs have to have dimension 3 (1st batch) or 2 (list of int lists)"

        num_qubits = self.cell.num_qubits
        input_size = len(inputs[0])
        input_lanes = range(input_size)

        psi = ket0(num_qubits)
        probs = []
        measured_seq = []

        for inpt, trgt in pairwise(inputs):
            assert (
                len(inpt) == input_size and len(trgt) == input_size
            ), "inputs all have to be the same length"

            p, psi = self.cell.forward(psi, inpt)
            probs.append(p)

            # measure output
            if postselect_measurement:
                measure = trgt
            else:
                output_distribution = torch.distributions.Categorical(probs=p)
                measure = tensor(
                    int_to_bin(output_distribution.sample(), width=input_size)
                )

            measured_seq.append(measure)
            psi = PostselectManyLayer(input_lanes, measure).forward(psi)

            # reset qubits
            psi = BitFlipLayer([i for i in input_lanes if measure[i]]).forward(psi)

        return torch.stack(probs), torch.stack(measured_seq)
