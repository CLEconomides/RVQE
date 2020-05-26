from ..data import *

import colorful


class DataMNISTBase(DataFactory):
    def __init__(self, shard: int, digits: List[int], scanlines: List[int], **kwargs):
        super().__init__(shard, **kwargs)

        import os.path as path
        import pandas as pd

        assert all(d in range(10) for d in digits), "digits have to be between 0 and 9"
        assert len(set(digits)) == len(digits), "duplicate digits"
        self.digits = digits

        assert all(s in range(3) for s in scanlines), "scanlines are 0, 1, or 2"
        assert len(set(scanlines)) == len(scanlines), "duplicate scanlines"
        self.scanlines = scanlines

        BIT_LUT = {n: tensor(int_to_bitword(n, 3))[scanlines].tolist() for n in range(8)}

        # import as list of lists of bitwords
        self._data = {
            digit: [
                [BIT_LUT[val.item()] for val in row]
                for row in tensor(
                    pd.read_csv(
                        path.join(
                            path.dirname(path.abspath(__file__)),
                            f"res/mnist-simple-{digit}-train.csv.gz",
                        ),
                        header=None,
                        compression="gzip",
                    ).values
                )
            ]
            for digit in digits
        }

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> tensor:
        return len(self.scanlines)

    def to_human(self, target: tensor, offset: int = 0) -> str:
        # the predicted image is missing the first pixel
        if len(target) == 99:
            target = torch.cat((tensor([[0, 0]]), target))

        # split scanlines
        targets = target.transpose(0, 1)

        # reshape to image
        targets = [t.reshape(10, 10) for t in targets]

        # group two rows per image
        targets = [[t.transpose(0, 1) for t in tgt.split(2)] for tgt in targets]

        # print
        PIXEL_REP = " ▄▀█"
        out = ""

        for lines in zip(*targets):
            out += "\t" + "".join(bitword_to_char(d, PIXEL_REP) for d in lines[0])
            for line in lines[1:]:
                out += "  " + "".join(bitword_to_char(d, PIXEL_REP) for d in line)
            out += "\n"

        return out[:-1]


class DataMNIST01(DataMNISTBase):
    """
        classify 0 and 1 on a 10x10 flattened input binary image of 0s and 1s
        The last pixel of the target contains the label.
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, digits=[0, 1], scanlines=[0, 1], **kwargs)

    # last pixel has to contain the label
    TARGET0 = torch.cat((torch.zeros(99, 2), torch.tensor([[0.0, 0]]))).int().tolist()
    TARGET1 = torch.cat((torch.zeros(99, 2), torch.tensor([[0.0, 1]]))).int().tolist()

    def next_batch(self) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []

        # we try to keep it balanced between 0 and 1, even if batch size is 1, and multiple shards are used
        dataA = self._data[0]
        dataB = self._data[1]
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
        if (
            offset == 0
            and not target.tolist() == DataMNIST01.TARGET0
            and not target.tolist() == DataMNIST01.TARGET1
        ):
            return super().to_human(target)
        else:
            return colorful.bold(bitword_to_str(target[-1]))

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

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            again we expect an input of length 99, so index 98 is the only one not ignored
        """
        return index != 98


class DataMNIST01_Gen(DataMNISTBase):
    """
        generate 10 x 10 binary images of 0s and 1s
        The first pixel is the label.
        As the first pixel is ignored in the prediction anyhow, there's no filtering.
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, digits=[0, 1], scanlines=[0, 1], **kwargs)

    def next_batch(self) -> Batch:
        # extract random batch of sentences
        sentences = []

        # we try to keep it balanced between 0 and 1, even if batch size is 1, and multiple shards are used
        dataA = self._data[0]
        dataB = self._data[1]
        labelA = [0, 0]
        labelB = [0, 1]
        if self.shard % 2 == 1:
            dataA, dataB = dataB, dataA
            labelA, labelB = labelB, labelA

        while len(sentences) < self.batch_size // 2:
            idx = torch.randint(0, len(dataA), (1,), generator=self.rng).item()
            sentence = dataA[idx].copy()
            sentence[0] = labelA
            sentences.append(sentence)

        while len(sentences) < self.batch_size:
            idx = torch.randint(0, len(dataB), (1,), generator=self.rng).item()
            sentence = dataB[idx].copy()
            sentence[0] = labelB
            sentences.append(sentence)

        # turn into batch
        return self._sentences_to_batch(sentences, targets=sentences)

    def to_human(self, target: tensor, offset: int = 0) -> str:
        return super().to_human(target)


class DataMNIST(DataMNISTBase):
    """
        classify the first eight MNIST digits on a 10x10 flattened input binary image of 0s and 1s
        The last two pixels (each two bits wide) of the target contains the label.
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, digits=[0, 1, 2, 3, 4, 5, 6, 7], scanlines=[0, 1, 2], **kwargs)

    LABELS = [
        [[1, 0, 0]], #0
        [[0, 1, 0]], #1
        [[0, 1, 1]], #2
        [[0, 0, 1]], #3

        [[0, 0, 0]], #4
        [[1, 0, 1]], #5
        [[1, 1, 1]], #6
        [[1, 1, 0]], #7
    ]

    # last two pixels has to contain the label
    TARGETS = [
        torch.cat((torch.zeros(99, 3), torch.tensor(label).float())).int().tolist()
        for label in LABELS
    ]

    def next_batch(self) -> Batch:
        # extract random batch of sentences
        sentences = []
        targets = []

        # we try to keep things balanced; so we fill them up as evenly randomly as possible
        while True:
            for digit_idx in torch.randperm(len(self.digits), generator=self.rng):
                digit = self.digits[digit_idx]
                data = self._data[digit]
                idx = torch.randint(0, len(data), (1,), generator=self.rng).item()
                sentences.append(data[idx])
                targets.append(DataMNIST.TARGETS[digit])

                # turn into batch
                if len(sentences) == self.batch_size:
                    return self._sentences_to_batch(sentences, targets)

    def to_human(self, target: tensor, offset: int = 0) -> str:
        if offset == 0 and not target.tolist() in DataMNIST.TARGETS:
            return super().to_human(target)
        else:
            label = target[-1:].tolist()
            return colorful.bold(
                "?" if not label in DataMNIST.LABELS else str(DataMNIST.LABELS.index(label))
            )

    def filter(self, sequence: tensor, dim: int) -> tensor:
        """
            we expect these to be offset by 1 from a proper output of length 100, i.e. only of length 99
            we only care about the last two pixel
        """
        assert sequence.dim() == 3 and dim in [1, 2]

        if dim == 1:
            return sequence[:, -1:, :]
        elif dim == 2:
            return sequence[:, :, -1:]

    def filter_sentence(self, sentence: tensor) -> tensor:
        return sentence[-1:]

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            again we expect an input of length 99, so index 97 and 98 are the only ones not ignored
        """
        return index not in [98]
