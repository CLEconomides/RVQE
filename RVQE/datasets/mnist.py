from ..data import *

import colorful


class DataMNIST01(DataFactory):
    _data = None

    @staticmethod
    def _load_mnist01():
        import os.path as path
        import pandas as pd

        DataMNIST01._data = {
            key: tensor(
                pd.read_csv(
                    path.join(path.dirname(path.abspath(__file__)), f"mnist-simple-{key}.csv")
                ).values
            )
            .unsqueeze(2)
            .tolist()
            for key in ["0-train", "1-train", "0-test", "1-test"]
        }

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)

        # local rng
        self.rng = torch.Generator().manual_seed(293784 + shard)

        if self._data == None:
            self._load_mnist01()

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> tensor:
        return 1

    # last pixel has to contain the label
    TARGET0 = torch.cat((torch.zeros(99), torch.zeros(1))).unsqueeze(1).int().tolist()
    TARGET1 = torch.cat((torch.zeros(99), torch.ones(1))).unsqueeze(1).int().tolist()

    def next_batch(self) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []

        # we try to keep it balanced between 0 and 1, even if batch size is 1, and multiple shards are used
        dataA = self._data["0-train"]
        dataB = self._data["1-train"]
        targetA = DataMNIST01.TARGET0
        targetB = DataMNIST01.TARGET1
        if self.shard % 2 == 1:
            dataA, dataB = dataB, dataA
            targetA, targetB = targetB, targetA

        while len(sentences) < self.batch_size // 2:
            idx = torch.randint(0, len(dataA), (1,), generator=self.rng).item()
            sentences.append(dataA[idx])
            targets.append(targetA)

        while len(sentences) < self.batch_size:
            idx = torch.randint(0, len(dataB), (1,), generator=self.rng).item()
            sentences.append(dataB[idx])
            targets.append(targetB)

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    def to_human(self, target: tensor, offset: int = 0) -> str:
        out = " " * offset + "".join(str(int(d)) for d in target[:-2])
        return out + "  " + colorful.bold(str(int(target[-1])))

    def filter(self, sequence: tensor, dim: int) -> tensor:
        """
            we expect these to be offset by 1 from a proper output of length 100, i.e. only of length 99
            we only care about the last pixel
        """
        assert sequence.dim() == 3 and dim in [1, 2]

        if dim == 1:
            return sequence[:, -1:, :]
        elif dim == 2:
            return sequence[:, :, -1:]

    def filter_sentence(self, sentence: tensor) -> tensor:
        return sentence[-1:]

    def ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            again we expect an input of length 99, so index 98 is the only one not ignored
        """
        return index != 98
