#!/usr/bin/env python

from typing import List, Union

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp
from timeit import default_timer as timer

from RVQE.model import RVQE
from RVQE.quantum import tensor
from RVQE import datasets

from math import pi

from termcolor import colored


class DistributedTrainingEnvironment:
    def __init__(self, shard: int, world_size: int, port: int):
        self.shard = shard
        self.world_size = world_size
        self.port = port

        print(
            f"[{shard}] Hello from shard {shard} in a world of size {world_size}! Happy training!"
        )

    def __enter__(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port)

        # initialize the process group
        torch.distributed.init_process_group("gloo", rank=self.shard, world_size=self.world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(6642)

        return self

    def __exit__(self, type, value, traceback):
        torch.distributed.destroy_process_group()

    def print_once(self, *args, **kwargs):
        if self.shard == 0:
            print(*args, **kwargs)

    def print_all(self, *args, **kwargs):
        print(f"[{self.shard}]", *args, **kwargs)

    def synchronize(self):
        return torch.distributed.barrier()

    def reduce(self, data: tensor, reduce_op: ReduceOp) -> tensor:
        torch.distributed.reduce(data, 0, reduce_op)
        return data

    def gather(self, data: tensor) -> List[tensor]:
        gather_list = None
        if self.shard == 0:
            gather_list = []
            for _ in range(self.world_size):
                gather_list.append(torch.zeros_like(data))

        torch.distributed.gather(data, gather_list, 0)
        return gather_list


def train(shard: int, args):
    with DistributedTrainingEnvironment(shard, args.num_shards, args.port) as environment:
        print, print_all = environment.print_once, environment.print_all

        EPOCHS = 10000

        if args.dataset == "simple":
            dataset = datasets.DataSimpleSentences(shard, **vars(args))
        elif args.dataset == "elman-xor":
            dataset =  datasets.DataElmanXOR(shard, **vars(args))
        elif args.dataset == "elman-letter":
            dataset =  datasets.DataElmanLetter(shard, **vars(args))
        elif args.dataset == "shakespeare":
            dataset =  datasets.DataShakespeare(**vars(args))

        # create model and distribute
        rvqe = RVQE(workspace_size=args.workspace, stages=args.stages, order=args.order)
        rvqe_ddp = DistributedDataParallel(rvqe)

        for p in rvqe_ddp.parameters():
            nn.init.uniform_(p, a=0.0, b=2 * pi)

        print(f"model has {len(list(rvqe_ddp.parameters()))} parameters.")

        postselected_training_phase = True
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(rvqe_ddp.parameters(), lr=args.learning_rate)
        elif args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(rvqe_ddp.parameters(), lr=args.learning_rate)

        criterion = nn.CrossEntropyLoss()

        time_start = timer()
        for epoch in range(EPOCHS):
            epoch += 1
            time_start = timer()

            # advance by one training batch
            optimizer.zero_grad()
            sentences, targets = dataset.next_batch()
            probs, _ = rvqe_ddp(sentences, postselect_measurement=True)
            if probs.dim() == 3:
                probs = probs.transpose(
                    1, 2
                )  # batch x classes x len  to match target with  batch x len
            loss = criterion(probs, targets[:, 1:])  # the model never predicts the first token
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
                    sentence, target = sentences[0], targets[0]
                    measured_probs, measured_seq = rvqe(sentence, postselect_measurement=False)
                    validation_loss = criterion(measured_probs, target[1:])
                    
                    # collect in main shard
                    sentences = environment.gather(sentence)
                    measured_seqs = environment.gather(measured_seq)
                    validation_loss = environment.reduce(validation_loss, ReduceOp.SUM)

                    if shard == 0:
                        assert len(measured_seqs) == args.num_shards, "gather failed somehow"

                        for i in range(min(3, args.num_shards)):
                            seq = measured_seqs[i]
                            sen = sentences[i]

                            print(
                                colored(f"gold = { dataset.to_human(sen) }", "yellow"),
                            )
                            print(
                                f"pred = { dataset.to_human(seq, offset=1) }"
                            )

                        validation_loss /= args.num_shards
                        print(colored(f"validation loss: {validation_loss:7.3e}", "green"))

                environment.synchronize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RVQE Training Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workspace", metavar="W", type=int, default=5, help="qubits to use as workspace",
    )
    parser.add_argument("--stages", metavar="S", type=int, default=2, help="RVQE cell stages")
    parser.add_argument(
        "--order", metavar="O", type=int, default=2, help="order of activation function"
    )
    parser.add_argument(
        "--port", metavar="P", type=int, default=12335, help="port for distributed computing",
    )
    parser.add_argument(
        "--num-shards",
        metavar="N",
        type=int,
        default=2,
        help="number of cores to use for parallel processing",
    )
    parser.add_argument(
        "--dataset",
        metavar="D",
        type=str,
        default="simple",
        help="dataset; choose between simple, xor and shakespeare",
    )
    parser.add_argument(
        "--sentence-length",
        metavar="SL",
        type=int,
        default=20,
        help="sentence length for data generators",
    )
    parser.add_argument(
        "--batch-size",
        metavar="B",
        type=int,
        default=1,
        help="batch size; only relevant for shakespeare dataset",
    )
    parser.add_argument(
        "--optimizer",
        metavar="OPT",
        type=str,
        default="adam",
        help="optimizer; one of adam or rmsprop",
    )
    parser.add_argument(
        "--learning-rate",
        metavar="LR",
        type=float,
        default="0.003",
        help="learning rate for optimizer",
    )

    args = parser.parse_args()

    # validate
    required_workspace = {"simple": 3, "elman-xor": 1, "elman-letter": 6, "shakespeare": 5}
    assert args.dataset in required_workspace, "invalid dataset"
    assert (
        required_workspace[args.dataset] < args.workspace
    ), f"need a workspace larger than {required_workspace[args.dataset]} for {args.dataset} dataset"
    assert args.optimizer in ["adam", "rmsprop"], "invalid optimizer"

    torch.multiprocessing.spawn(train, args=(args,), nprocs=args.num_shards, join=True)
