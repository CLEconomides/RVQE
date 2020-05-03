#!/usr/bin/env python

from typing import List, Union

import os, time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer

from RVQE.model import RVQE
from RVQE.quantum import tensor
from RVQE import datasets

from math import pi

from termcolor import colored

import secrets


class MockSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_hparams(self, *args, **kwargs):
        pass


class DistributedTrainingEnvironment:
    def __init__(self, shard: int, args):
        self.shard = shard
        self.world_size = args.num_shards
        self.port = args.port
        self._original_args = args
        # the hex tokens are different in different shards; so checkpoint from the same shard always
        # this has to be set only initially, as it'll be restored on resume
        if hasattr(args, "dataset"):
            self._checkpoint_prefix = f"-{args.tag}-{args.dataset}--{secrets.token_hex(3)}"

        print(f"[{shard}] Hello from shard {shard} in a world of size {self.world_size}! Happy training!")

    def __enter__(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port)

        # initialize the process group
        torch.distributed.init_process_group("gloo", rank=self.shard, world_size=self.world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(7856)

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

    CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/")

    def save_checkpoint(self, model, optimizer, extra_tag: str = "", **kwargs):
        """
            we don't check whether this is only called from shard 0
        """
        kwargs.update(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "_original_args": self._original_args,
                "_checkpoint_prefix": self._checkpoint_prefix,
                "_torch_rng_state": torch.get_rng_state(),
            }
        )

        filename = f"checkpoint-{self._checkpoint_prefix}-{extra_tag}-" + time.strftime("%Y-%m-%d--%H-%M-%S") + ".tar"
        path = os.path.join(self.CHECKPOINT_PATH, filename)

        torch.save(kwargs, path)
        return filename

    def load_checkpoint(self, path: str) -> dict:
        store = torch.load(path)

        self._original_args = store["_original_args"]
        self._checkpoint_prefix = store["_checkpoint_prefix"]
        torch.set_rng_state(store["_torch_rng_state"])

        return store

    @property
    def logger(self) -> SummaryWriter:
        if self.shard != 0:
            self._logger = MockSummaryWriter()

        if not hasattr(self, "_logger"):
            self._logger = SummaryWriter(comment=f"{self._checkpoint_prefix}")

        return self._logger


def dict_to_table(dct) -> str:
    return "\r".join(f"    {k:>25} {v}" for k, v in dct.items())


def train(shard: int, args):
    with DistributedTrainingEnvironment(shard, args) as environment:
        print, print_all = environment.print_once, environment.print_all

        RESUME_MODE = hasattr(args, "filename")

        # either load or initialize new
        if RESUME_MODE:
            store = environment.load_checkpoint(args.filename)
            original_args = store["_original_args"]
            epoch_start = store["epoch"]
            best_validation_loss = store["best_validation_loss"]
            best_character_error_rate = store["best_character_error_rate"]
        else:
            original_args = args
            epoch_start = 0
            best_validation_loss = None
            best_character_error_rate = None

        environment.logger.add_text("args", dict_to_table(vars(args)), epoch_start)

        if original_args.dataset == "simple-seq":
            dataset = datasets.DataSimpleSequences(shard, **vars(original_args))
        if original_args.dataset == "simple-quotes":
            dataset = datasets.DataSimpleQuotes(shard, **vars(original_args))
        elif original_args.dataset == "elman-xor":
            dataset = datasets.DataElmanXOR(shard, **vars(original_args))
        elif original_args.dataset == "elman-letter":
            dataset = datasets.DataElmanLetter(shard, **vars(original_args))
        elif original_args.dataset == "shakespeare":
            dataset = datasets.DataShakespeare(**vars(original_args))

        # create model and distribute
        rvqe = RVQE(workspace_size=original_args.workspace, stages=original_args.stages, order=original_args.order,)
        rvqe_ddp = DistributedDataParallel(rvqe)

        # create optimizer
        if original_args.optimizer == "sgd":
            optimizer = torch.optim.SGD(rvqe_ddp.parameters(), lr=original_args.learning_rate)
        elif original_args.optimizer == "adam":
            optimizer = torch.optim.Adam(rvqe_ddp.parameters(), lr=original_args.learning_rate)
        elif original_args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(rvqe_ddp.parameters(), lr=original_args.learning_rate)
        elif original_args.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(rvqe_ddp.parameters(), lr=original_args.learning_rate)

        # when in resume mode, load model and optimizer state; otherwise initialize
        if RESUME_MODE:
            rvqe_ddp.load_state_dict(store["model_state_dict"])
            optimizer.load_state_dict(store["optimizer_state_dict"])
        else:
            for p in rvqe_ddp.parameters():
                nn.init.uniform_(p, a=0.0, b=2 * pi)

        # cross entropy loss
        criterion = nn.CrossEntropyLoss()

        # wait for all shards to be happy
        environment.synchronize()

        if RESUME_MODE:
            print(
                f"⏩  Resuming session! Model has {len(list(rvqe_ddp.parameters()))} parameters, and we start at epoch {epoch_start} with best validation loss {best_validation_loss:7.3e}."
            )
        else:
            print(f"⏩  New session! Model has {len(list(rvqe_ddp.parameters()))} parameters.")

        time_start = timer()
        for epoch in range(epoch_start, args.epochs):
            epoch += 1
            time_start = timer()

            # advance by one training batch
            loss = None
            sentences, targets = dataset.next_batch()

            def loss_closure():
                nonlocal loss  # write to loss outside closure
                nonlocal targets

                optimizer.zero_grad()
                probs, _ = rvqe_ddp(sentences, postselect_measurement=True)
                _probs, _targets = dataset.filter(probs, targets[:, 1:])
                loss = criterion(_probs, _targets)  # the model never predicts the first token
                loss.backward()

                return loss

            optimizer.step(loss_closure)

            # print loss each few epochs
            if epoch % 1 == 0:
                print(f"{epoch:04d}/{args.epochs:04d} {timer() - time_start:5.1f}s  loss={loss:7.3e}")

            # log
            environment.logger.add_scalar("loss/train", loss, epoch)
            environment.logger.add_scalar("time", timer() - time_start, epoch)

            # print samples every few epochs or the last round
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                with torch.no_grad():
                    # run entire batch through the network without postselecting measurements
                    measured_probs, measured_seqs = rvqe(sentences, postselect_measurement=False)
                    _probs, _targets = dataset.filter(measured_probs, targets[:, 1:])
                    validation_loss = criterion(_probs, _targets)

                    # collect in main shard
                    sentences = environment.gather(sentences)
                    measured_seqs = environment.gather(measured_seqs)
                    validation_loss = environment.reduce(validation_loss, ReduceOp.SUM)

                    if shard == 0:
                        sentences = torch.cat(sentences)
                        measured_seqs = torch.cat(measured_seqs)
                        validation_loss /= args.num_shards

                        assert len(measured_seqs) == args.num_shards * args.batch_size, "gather failed somehow"

                        logtext = ""
                        for i in range(min(args.num_validation_samples, args.num_shards)):
                            seq = measured_seqs[i]
                            sen = sentences[i]

                            text = f"gold = { dataset.to_human(sen) }"
                            print(colored(text, "yellow"),)
                            logtext += "    " + text + "\r\n"
                            text = f"pred = { dataset.to_human(seq, offset=1) }"
                            print(text)
                            logtext += "    " + text + "\r\n"


                        # character error rate
                        total = 0
                        correct = 0
                        for seq, sen in zip(measured_seqs, sentences):
                            seq = dataset.filter_sentence(seq)
                            sen = dataset.filter_sentence(sen[1:])
                            for wa, wb in zip(seq, sen):
                                total += 1
                                correct += 1 if torch.all(wa == wb) else 0

                        character_error_rate = 1 - correct / total

                        print(colored(f"validation loss:       {validation_loss:7.3e}", "green"))
                        print(colored(f"character error rate:  {character_error_rate:.3f}", "green"))

                        # log
                        environment.logger.add_scalar("loss/validate", validation_loss, epoch)
                        environment.logger.add_scalar("accuracy/cer_current", character_error_rate, epoch)
                        environment.logger.add_text("validation_samples", logtext, epoch)

                        if best_character_error_rate is None or character_error_rate < best_character_error_rate:
                            best_character_error_rate = character_error_rate
                            environment.logger.add_scalar("accuracy/cer_best", best_character_error_rate, epoch)

                        # checkpointing
                        if best_validation_loss is None or validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            environment.logger.add_scalar("loss/validate_best", best_validation_loss, epoch)
                            checkpoint = environment.save_checkpoint(
                                rvqe_ddp,
                                optimizer,
                                **{
                                    "epoch": epoch,
                                    "best_validation_loss": best_validation_loss,
                                    "best_character_error_rate": best_character_error_rate,
                                },
                            )
                            if checkpoint is not None:
                                environment.logger.add_text("checkpoint", checkpoint, epoch)
                                print(f"saved new best checkpoint {checkpoint}")

                        # END shard 0 tasks
                    # END validation

                environment.synchronize()
                # END training loop

        # Training done
        checkpoint = environment.save_checkpoint(
            rvqe_ddp, optimizer, extra_tag="final", **{"epoch": epoch, "best_validation_loss": best_validation_loss},
        )
        environment.logger.add_hparams(
            {k: v for k, v in vars(original_args).items() if isinstance(v, (int, float, str, bool, torch.Tensor))},
            {
                "hparams/epoch": epoch,
                "hparams/num_parameters": len(list(rvqe_ddp.parameters())),
                "hparams/validate_best": best_validation_loss,
                "hparams/cer_best": best_character_error_rate,
            },
        )
        print(f"⏩  DONE. Written final checkpoint to {checkpoint}")


def command_train(args):
    # validate
    required_workspace = {
        "simple-seq": 3,
        "simple-quotes": 5,
        "elman-xor": 1,
        "elman-letter": 6,
        "shakespeare": 5,
    }
    assert args.dataset in required_workspace, "invalid dataset"
    assert (
        required_workspace[args.dataset] < args.workspace
    ), f"need a workspace larger than {required_workspace[args.dataset]} for {args.dataset} dataset"
    assert args.optimizer in ["sgd", "adam", "rmsprop", "lbfgs"], "invalid optimizer"

    torch.multiprocessing.spawn(train, args=(args,), nprocs=args.num_shards, join=True)


def command_resume(args):
    torch.multiprocessing.spawn(train, args=(args,), nprocs=args.num_shards, join=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RVQE Training Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port", metavar="P", type=int, default=12335, help="port for distributed computing",
    )
    parser.add_argument(
        "--num-shards", metavar="N", type=int, default=2, help="number of cores to use for parallel processing",
    )
    parser.add_argument(
        "--num-validation-samples",
        metavar="VALS",
        type=int,
        default=2,
        help="number of validation samples to draw each 10 epochs",
    )
    parser.add_argument("--tag", metavar="TAG", type=str, default="", help="tag for checkpoints and logs")
    parser.add_argument("--epochs", metavar="EP", type=int, default=5000, help="number of learning epochs")

    subparsers = parser.add_subparsers(help="available commands")

    parser_train = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser_train.set_defaults(func=command_train)
    parser_train.add_argument(
        "--workspace", metavar="W", type=int, default=5, help="qubits to use as workspace",
    )
    parser_train.add_argument("--stages", metavar="S", type=int, default=2, help="RVQE cell stages")
    parser_train.add_argument("--order", metavar="O", type=int, default=2, help="order of activation function")
    parser_train.add_argument(
        "--dataset",
        metavar="D",
        type=str,
        default="simple-seq",
        help="dataset; choose between simple-seq, simple-quotes, elman-xor, elman-letter and shakespeare",
    )
    parser_train.add_argument(
        "--sentence-length", metavar="SL", type=int, default=20, help="sentence length for data generators",
    )
    parser_train.add_argument(
        "--batch-size", metavar="B", type=int, default=1, help="batch size",
    )
    parser_train.add_argument(
        "--optimizer", metavar="OPT", type=str, default="adam", help="optimizer; one of sgd, adam or rmsprop",
    )
    parser_train.add_argument(
        "--learning-rate", metavar="LR", type=float, default="0.003", help="learning rate for optimizer",
    )

    parser_resume = subparsers.add_parser("resume", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser_resume.set_defaults(func=command_resume)
    parser_resume.add_argument("filename", type=str, help="checkpoint filename")

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)
