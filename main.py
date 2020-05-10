#!/usr/bin/env python

from typing import List, Union, Optional

import os, time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer

from RVQE.model import RVQE, count_parameters
from RVQE.quantum import tensor
from RVQE import datasets, data

import math, re

import colorful

import secrets


# colorful printing

colorful.use_palette(
    {
        "background": "#0B2A71",
        "white": "#ffffff",
        "gold": "#EDC835",
        "validate": "#7EBE7B",
        "faint": "#606060",
    }
)


def colorless(line: colorful.core.ColorfulString) -> str:
    while isinstance(line, colorful.core.ColorfulString):
        line = line.orig_string
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


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
        self.seed = args.seed
        self.timeout = args.timeout
        self._time_start = timer()
        self._original_args = args
        # the hex tokens are different in different shards; so checkpoint from the same shard always
        # this has to be set only initially, as it'll be restored on resume
        if hasattr(args, "dataset"):
            self._checkpoint_prefix = f"-{args.tag}-{args.dataset}--{secrets.token_hex(3)}"

        print(
            f"[{shard}] Hello from shard {shard} in a world of size {self.world_size}! Happy training!"
        )

    def __enter__(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port)

        # initialize the process group
        torch.distributed.init_process_group("gloo", rank=self.shard, world_size=self.world_size)

        # Explicitly setting seed to make sure that models created in two processes
        # start from same random weights and biases.
        torch.manual_seed(self.seed)

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

    @property
    def is_timeout(self) -> bool:
        if self.timeout is None:
            return False

        ret = tensor([0])
        if (timer() - self._time_start) > self.timeout:
            ret[:] = 1
            torch.distributed.broadcast(ret, 0)

        self.synchronize()

        return ret.item() == 1

    CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/")

    def save_checkpoint(self, model, optimizer, extra_tag: str = "", **kwargs) -> Optional[str]:
        if self.shard != 0:
            return None

        kwargs.update(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "_original_args": self._original_args,
                "_checkpoint_prefix": self._checkpoint_prefix,
                "_torch_rng_state": torch.get_rng_state(),
            }
        )

        filename = (
            f"checkpoint-{self._checkpoint_prefix}-{extra_tag}-"
            + time.strftime("%Y-%m-%d--%H-%M-%S")
            + ".tar"
        )
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
        rvqe = DistributedDataParallel(
            RVQE(
                workspace_size=original_args.workspace,
                input_size=dataset.input_width,
                stages=original_args.stages,
                order=original_args.order,
                degree=original_args.degree,
            )
        )

        # create optimizer
        if original_args.optimizer == "sgd":
            optimizer = torch.optim.SGD(rvqe.parameters(), lr=original_args.learning_rate)
        elif original_args.optimizer == "adam":
            optimizer = torch.optim.AdamW(rvqe.parameters(), lr=original_args.learning_rate)
        elif original_args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(rvqe.parameters(), lr=original_args.learning_rate)
        elif original_args.optimizer == "lbfgs":
            optimizer = torch.optim.LBFGS(rvqe.parameters(), lr=original_args.learning_rate)

        # when in resume mode, load model and optimizer state; otherwise initialize
        if RESUME_MODE:
            rvqe.load_state_dict(store["model_state_dict"])
            optimizer.load_state_dict(store["optimizer_state_dict"])
        else:
            for name, p in rvqe.named_parameters():
                if name[-1:] == "θ":  # rY
                    pass
                    # nn.init.normal_(p, mean=0.0, std=0.005)
                elif name[-1:] == "φ":  # crY
                    nn.init.normal_(p, mean=0.0, std=0.5)
                else:
                    raise NotImplementedError(f"{name} unknown parameter name for initialization")

        # cross entropy loss
        _criterion = nn.CrossEntropyLoss()
        BEST_LOSS_POSSIBLE = -1 + math.log(
            2 ** dataset.input_width - 1 + math.e
        )  # see formula for CrossEntropyLoss
        criterion = lambda *args, **kwargs: _criterion(*args, **kwargs) - BEST_LOSS_POSSIBLE
        print(
            colorful.validate(f"best possible loss: {BEST_LOSS_POSSIBLE:7.3e}", "magenta"),
            "automatically subtracted",
        )

        # wait for all shards to be happy
        environment.synchronize()

        if RESUME_MODE:
            print(
                f"🔄  Resuming session! Model has {count_parameters(rvqe)} parameters, and we start at epoch {epoch_start} with best validation loss {best_validation_loss:7.3e}."
            )
        else:
            print(f"⏩  New session! Model has {count_parameters(rvqe)} parameters.")

        for epoch in range(epoch_start, args.epochs):
            # check if we should timeout
            if environment.is_timeout:
                print(f"❎  Timeout hit after {original_args.timeout}s.")
                break

            time_start = timer()
            # advance by one training batch
            loss = None
            sentences, targets = dataset.next_batch()

            def loss_closure():
                nonlocal loss  # write to loss outside closure
                nonlocal targets

                optimizer.zero_grad()
                probs, _ = rvqe(sentences, targets, postselect_measurement=True)
                _probs = dataset.filter(probs, dim=2)
                _targets = dataset.filter(data.skip_first(targets), dim=1)
                loss = criterion(
                    _probs, data.targets_for_loss(_targets)
                )  # the model never predicts the first token
                loss.backward()

                return loss

            optimizer.step(loss_closure)

            # print loss each few epochs
            if epoch % 1 == 0:
                print(
                    f"{epoch:04d}/{args.epochs:04d} {timer() - time_start:5.1f}s  loss={loss:7.3e}"
                )

            # log
            environment.logger.add_scalar("loss/train", loss, epoch)
            environment.logger.add_scalar("time", timer() - time_start, epoch)

            # print samples every few epochs or the last round
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                with torch.no_grad():
                    # run entire batch through the network without postselecting measurements
                    measured_probs, measured_sequences = rvqe(
                        sentences, targets, postselect_measurement=dataset.ignore_output_at_step
                    )
                    _probs = dataset.filter(measured_probs, dim=2)
                    _targets = dataset.filter(data.skip_first(targets), dim=1)
                    validation_loss = criterion(_probs, data.targets_for_loss(_targets))

                    # collect in main shard
                    sentences = environment.gather(sentences)
                    targets = environment.gather(targets)
                    measured_sequences = environment.gather(measured_sequences)
                    validation_loss = environment.reduce(validation_loss, ReduceOp.SUM)

                    if shard == 0:
                        sentences = torch.cat(sentences)
                        targets = torch.cat(targets)
                        measured_sequences = torch.cat(measured_sequences)
                        validation_loss /= args.num_shards

                        assert (
                            len(measured_sequences) == args.num_shards * args.batch_size
                        ), "gather failed somehow"

                        logtext = ""
                        for i in range(min(args.num_validation_samples, args.num_shards)):
                            if (targets[i] != sentences[i]).any():
                                text = f"inpt = { dataset.to_human(sentences[i]) }"
                                print(colorful.faint(text))
                                logtext += "    " + text + "\r\n"
                            text = f"gold = { dataset.to_human(targets[i]) }"
                            print(colorful.gold(text))
                            logtext += "    " + colorless(text) + "\r\n"
                            text = f"pred = { dataset.to_human(measured_sequences[i], offset=1) }"
                            print(text)
                            logtext += "    " + colorless(text) + "\r\n"

                        # character error rate
                        character_error_rate = data.character_error_rate(
                            dataset.filter(measured_sequences, dim=1),
                            dataset.filter(data.skip_first(targets), dim=1)
                        )

                        print(
                            colorful.bold_validate(
                                f"validation loss:       {validation_loss:7.3e}", "green"
                            )
                        )
                        print(
                            colorful.validate(
                                f"character error rate:  {character_error_rate:.3f}", "green"
                            )
                        )

                        # log
                        environment.logger.add_scalar("loss/validate", validation_loss, epoch)
                        environment.logger.add_scalar(
                            "accuracy/character_error_rate_current", character_error_rate, epoch
                        )
                        environment.logger.add_text("validation_samples", logtext, epoch)

                        if (
                            best_character_error_rate is None
                            or character_error_rate < best_character_error_rate
                        ):
                            best_character_error_rate = character_error_rate
                            environment.logger.add_scalar(
                                "accuracy/character_error_rate_best",
                                best_character_error_rate,
                                epoch,
                            )

                        # checkpointing
                        if best_validation_loss is None or validation_loss < best_validation_loss:
                            best_validation_loss = validation_loss
                            environment.logger.add_scalar(
                                "loss/validate_best", best_validation_loss, epoch
                            )
                            checkpoint = environment.save_checkpoint(
                                rvqe,
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

                    # ENDIF shard 0 tasks

                # ENDWITH torch.no_grad

                if args.stop_at_loss is not None and args.stop_at_loss > validation_loss:
                    break  # breaks out of training loop

            # ENDIF validation

            environment.synchronize()
        # END training loop

        # Training done
        checkpoint = environment.save_checkpoint(
            rvqe,
            optimizer,
            extra_tag="final" if not environment.is_timeout else "interrupted",
            **{"epoch": epoch, "best_validation_loss": best_validation_loss},
        )
        environment.logger.add_hparams(
            {
                k: v
                for k, v in vars(original_args).items()
                if isinstance(v, (int, float, str, bool, torch.Tensor))
            },
            {
                "hparams/epoch": epoch,
                "hparams/num_parameters": len(list(rvqe.parameters())),
                "hparams/validate_best": best_validation_loss,
                "hparams/character_error_rate_best": best_character_error_rate,
            },
        )
        print(f"🆗  DONE. Written final checkpoint to {checkpoint}")


def command_train(args):
    # validate
    datasets = {
        "simple-seq",
        "simple-quotes",
        "elman-xor",
        "elman-letter",
        "shakespeare",
    }
    assert args.dataset in datasets, "invalid dataset"
    assert args.optimizer in {"sgd", "adam", "rmsprop", "lbfgs"}, "invalid optimizer"

    if args.dataset == "simple-seq":
        assert (
            args.num_shards == 2
            and args.batch_size == 1
            or args.num_shards == 1
            and args.batch_size == 2
        )
    if args.dataset == "simple-quotes":
        assert (
            args.num_shards == 5
            and args.batch_size == 1
            or args.num_shards == 1
            and args.batch_size == 5
        )

    if args.num_shards == 1:
        train(0, args)
    else:
        torch.multiprocessing.spawn(train, args=(args,), nprocs=args.num_shards, join=True)


def command_resume(args):
    torch.multiprocessing.spawn(train, args=(args,), nprocs=args.num_shards, join=True)


if __name__ == "__main__":

    title = " RVQE Trainer "
    print(
        colorful.background("▄" * len(title))
        + "\n"
        + colorful.bold_white_on_background(title)
        + "\n"
        + colorful.background("▀" * len(title))
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="RVQE Training Script", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--num-validation-samples",
        metavar="VALS",
        type=int,
        default=2,
        help="number of validation samples to draw each 10 epochs",
    )
    parser.add_argument(
        "--tag", metavar="TAG", type=str, default="", help="tag for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs", metavar="EP", type=int, default=5000, help="number of learning epochs"
    )
    parser.add_argument(
        "--timeout",
        metavar="TO",
        type=int,
        default=None,
        help="timeout in s after what time to interrupt",
    )
    parser.add_argument(
        "--stop-at-loss",
        metavar="SL",
        type=float,
        default=None,
        help="stop at this validation loss",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=82727,
        help="random seed for parameter initialization",
    )

    subparsers = parser.add_subparsers(help="available commands")

    parser_train = subparsers.add_parser(
        "train", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_train.set_defaults(func=command_train)
    parser_train.add_argument(
        "--workspace", metavar="W", type=int, default=3, help="qubits to use as workspace",
    )
    parser_train.add_argument("--stages", metavar="S", type=int, default=2, help="RVQE cell stages")
    parser_train.add_argument(
        "--order", metavar="O", type=int, default=2, help="order of activation function"
    )
    parser_train.add_argument(
        "--degree", metavar="O", type=int, default=2, help="degree of quantum neuron"
    )
    parser_train.add_argument(
        "--dataset",
        metavar="D",
        type=str,
        default="simple-seq",
        help="dataset; choose between simple-seq, simple-quotes, elman-xor, elman-letter and shakespeare",
    )
    parser_train.add_argument(
        "--sentence-length",
        metavar="SL",
        type=int,
        default=20,
        help="sentence length for data generators",
    )
    parser_train.add_argument(
        "--batch-size", metavar="B", type=int, default=1, help="batch size",
    )
    parser_train.add_argument(
        "--optimizer",
        metavar="OPT",
        type=str,
        default="rmsprop",
        help="optimizer; one of sgd, adam or rmsprop",
    )
    parser_train.add_argument(
        "--learning-rate",
        metavar="LR",
        type=float,
        default="0.003",
        help="learning rate for optimizer",
    )

    parser_resume = subparsers.add_parser(
        "resume", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_resume.set_defaults(func=command_resume)
    parser_resume.add_argument("filename", type=str, help="checkpoint filename")

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        args.func(args)
