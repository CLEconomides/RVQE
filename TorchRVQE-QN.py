#!/usr/bin/env python

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from timeit import default_timer as timer

from RVQE.model import RVQE
from RVQE.quantum import tensor
from RVQE.data import *

from math import pi


class DistributedTrainingEnvironment:
    def __init__(self, shard: int, world_size: int):
        self.shard = shard
        self.world_size = world_size

        print(
            f"Hello from shard {shard} in a world of size {world_size}! Happy training!"
        )

    def __enter__(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        torch.distributed.init_process_group(
            "gloo", rank=self.shard, world_size=self.world_size
        )

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(6642)

        return self

    def __exit__(self, type, value, traceback):
        torch.distributed.destroy_process_group()

    def print_once(self, *args, **kwargs):
        if self.shard == 0:
            print(*args, **kwargs)


def train(shard: int, world_size: int):
    with DistributedTrainingEnvironment(shard, world_size) as environment:
        print = environment.print_once

        EPOCHS = 10000
        SENTENCE_LENGTH = 20
        INPUT_WIDTH = 3
        DATASET = tensor(
            [
                alternating_sentence(
                    SENTENCE_LENGTH, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
                ),
                constant_sentence(SENTENCE_LENGTH, [1, 0, 0]),
            ]
        )
        TARGETS = tensor(
            [
                [
                    bin_to_label(word)
                    for word in sentence[
                        1:
                    ]  # since we want to predict the next character at all times
                ]
                for sentence in DATASET
            ]
        )

        # batch dataset and target - for now just one batch of length 2
        DATASET = DATASET.unsqueeze(0)
        TARGETS = TARGETS.unsqueeze(0)

        # create model and distribute
        rvqe = RVQE(workspace_size=5, stages=2, order=2)
        rvqe_ddp = DistributedDataParallel(rvqe)

        for p in rvqe_ddp.parameters():
            nn.init.uniform_(p, a=0.0, b=2 * pi)

        print(f"model has {len(list(rvqe_ddp.parameters()))} parameters.")

        postselected_training_phase = True
        optimizer = torch.optim.Adam(rvqe_ddp.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()

        time_start = timer()
        for epoch in range(EPOCHS):
            epoch += 1
            time_start = timer()

            optimizer.zero_grad()

            def run(sentence, target):
                probs, _ = rvqe_ddp(sentence, postselect_measurement=True)
                if probs.dim() == 3:
                    probs = probs.transpose(
                        1, 2
                    )  # batch x classes x len  to match target with  batch x len
                loss = criterion(probs, target)
                return loss

            loss = torch.stack(
                [
                    run(sentence[shard], target[shard])
                    for sentence, target in zip(DATASET, TARGETS)
                ]
            ).mean()
            loss.backward()
            optimizer.step()

            # print loss each few epochs
            if epoch % 1 == 0:
                print(
                    f"{'PS' if postselected_training_phase else 'MS'} {epoch:04d}/{EPOCHS:04d} {timer() - time_start:5.1f}s  loss={loss:7.3e}"
                )

            # print samples every few epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    for sentence in DATASET[0]:
                        _, measured_seq = rvqe(sentence, postselect_measurement=False)
                        print(
                            "\x1b[33mgold =",
                            " ".join([str(bin_to_label(word)) for word in sentence]),
                            "\x1b[0m",
                        )
                        print(
                            "pred =",
                            " ",  # we don't predict the first word
                            " ".join(
                                [str(bin_to_label(word)) for word in measured_seq]
                            ),
                        )


if __name__ == "__main__":
    WORLD_SIZE = 1
    pc = torch.multiprocessing.spawn(
        train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True
    )
