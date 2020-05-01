from ..data import *


def constant_sentence(length: int, constant: Bitword) -> List[Bitword]:
    return [constant for _ in range(length)]


def alternating_sentence(length: int, constants: List[Bitword]) -> List[Bitword]:
    return [constants[i % len(constants)] for i in range(length)]



class DataSimpleSentences(DataFactory):
    def __init__(self, shard: int, num_shards: int, **kwargs):
        kwargs.update({ "batch_size": max(1, 2 // num_shards) })
        super().__init__(shard, num_shards=num_shards, **kwargs)

        sentences = [
            alternating_sentence(self.sentence_length, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
            constant_sentence(self.sentence_length, [1, 0, 0]),
        ]

        self._batches_data = self._sentences_to_batches(sentences)

    @property
    def _batches(self) -> List[Batch]:
        return self._batches_data

    @property
    def input_width(self) -> tensor:
        return 3

    def to_human(self, target: tensor, offset: int = 0) -> str:
        return "  "*offset + " ".join([ str(bitword_to_int(word)) for word in target])

