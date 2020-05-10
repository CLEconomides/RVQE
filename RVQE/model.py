from typing import Tuple, List, Dict, Optional, Union, Callable

from .compound_layers import (
    UnitaryLayer,
    QuantumNeuronLayer,
    FastQuantumNeuronLayer,
    BitFlipLayer,
    PostselectManyLayer,
)
from .quantum import tensor, ket0, probabilities, num_state_qubits
from .data import zip_with_offset, int_to_bitword, Bitword

import torch
from torch import nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RVQECell(nn.Module):
    def __init__(
        self,
        *,
        workspace_size: int,
        input_size: int,
        stages: int,
        order: int = 2,
        degree: int = 2,
        fast: bool = True,
    ):
        """
            by default we set up the qubit indices to be
            input, workspace, ancillas
        """
        assert (
            workspace_size >= 1 and input_size >= 1 and stages >= 1 and order >= 1
        ), "all parameters have to be >= 1"

        super().__init__()

        self.stages = stages
        self.order = order
        self.degree = degree

        QNL_T = FastQuantumNeuronLayer if fast else QuantumNeuronLayer
        ancilla_count = QNL_T.ancillas_for_order(order)

        self.inout = list(range(0, input_size))
        self.workspace = list(range(input_size, input_size + workspace_size))
        self.ancillas = list(
            range(input_size + workspace_size, input_size + workspace_size + ancilla_count)
        )

        self.input_layer = nn.Sequential(
            *[
                QNL_T(
                    workspace=self.workspace + self.inout,
                    outlane=out,
                    order=order,
                    ancillas=self.ancillas,
                    degree=degree,
                )
                for out in self.workspace
            ]
        )
        self.kernel_layer = nn.Sequential(
            *[
                nn.Sequential(
                    UnitaryLayer(self.workspace),
                    *[
                        QNL_T(
                            workspace=self.workspace + self.inout,
                            outlane=out,
                            order=order,
                            ancillas=self.ancillas,
                        )
                        for out in self.workspace
                    ],
                )
                for _ in range(stages)
            ]
        )
        self.output_layer = nn.Sequential(
            *[
                QNL_T(
                    workspace=self.workspace + self.inout,
                    outlane=out,
                    order=order,
                    ancillas=self.ancillas,
                )
                for out in self.inout
            ]
        )

    @property
    def num_qubits(self) -> int:
        return len(self.ancillas) + len(self.workspace) + len(self.inout)

    def forward(self, psi: tensor, input: Bitword) -> Tuple[tensor, tensor]:
        # we assume psi has its input lanes reset to 0
        assert len(input) == len(self.inout), "wrong input size given"
        assert num_state_qubits(psi) == self.num_qubits, "state given does not have the right size"

        bitflip_layer = self.bitflip_for(input)

        # input and kernel layers don't write to the inout lanes, it is read only
        psi = bitflip_layer.forward(psi)
        psi = self.input_layer.forward(psi)
        psi = self.kernel_layer.forward(psi)

        # reset inout lanes to 000, then write output
        psi = bitflip_layer.forward(psi)
        psi = self.output_layer.forward(psi)

        return probabilities(psi, self.inout), psi

    def bitflip_for(self, input: Bitword) -> BitFlipLayer:
        return BitFlipLayer([lane for i, lane in enumerate(self.inout) if input[i] == 1])


class RVQE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cell = RVQECell(**kwargs)

    def forward(
        self,
        inputs: tensor,
        targets: tensor,
        postselect_measurement: Union[bool, Callable[[int], bool]],
    ) -> Tuple[tensor, list]:
        if isinstance(postselect_measurement, bool):
            # return callback that gives constant
            _postselect_measurement = postselect_measurement
            postselect_measurement = lambda _, __: _postselect_measurement

        # batched call; return stacked result
        if inputs.dim() == 3:
            assert targets.dim() == 3, "inputs have dimension 3, but targets not"
            batch_measured_seq = []
            batch_probs = []
            for inpt, trgt in zip(inputs, targets):
                probs, measured_seq = self.forward(inpt, trgt, postselect_measurement)
                batch_probs.append(probs)
                batch_measured_seq.append(measured_seq)

            # we transpose the batch_probs such that the len dimension comes last
            return torch.stack(batch_probs).transpose(1, 2), torch.stack(batch_measured_seq)

        # normal call
        assert (
            inputs.dim() == 2 and targets.dim() == 2
        ), "inputs have to have dimension 3 (1st batch) or 2 (list of int lists)"

        psi = ket0(self.cell.num_qubits)
        probs = []
        measured_seq = []

        for i, (inpt, trgt) in enumerate(zip_with_offset(inputs, targets)):
            p, psi = self.cell.forward(psi, inpt)
            probs.append(p)

            # measure output
            if postselect_measurement(i, trgt):
                measure = trgt
            else:
                output_distribution = torch.distributions.Categorical(probs=p)
                measure = tensor(
                    int_to_bitword(output_distribution.sample(), width=len(self.cell.inout))
                )

            measured_seq.append(measure)
            psi = PostselectManyLayer(self.cell.inout, measure).forward(psi)

            # reset qubits
            psi = self.cell.bitflip_for(measure).forward(psi)

        probs = torch.stack(probs)
        measured_seq = torch.stack(measured_seq)
        return probs, measured_seq
