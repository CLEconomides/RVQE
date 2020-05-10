from ..data import *


def constant_sentence(length: int, constant: Bitword) -> List[Bitword]:
    return [constant for _ in range(length)]


def alternating_sentence(length: int, constants: List[Bitword]) -> List[Bitword]:
    return [constants[i % len(constants)] for i in range(length)]


class DataSimpleSequences(DataFactory):
    def __init__(self, shard: int, num_shards: int, **kwargs):
        kwargs.update({"batch_size": max(1, 2 // num_shards)})
        super().__init__(shard, num_shards=num_shards, **kwargs)

        sentences = [
            alternating_sentence(
                self.sentence_length, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
            ),
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
        return "  " * offset + " ".join([str(bitword_to_int(word)) for word in target])


class DataSimpleQuotes(DataFactory):
    VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! \n"
    DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! ¶"
    assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

    def __init__(self, shard: int, num_shards: int, **kwargs):
        kwargs.update({"batch_size": max(1, 5 // num_shards)})
        super().__init__(shard, num_shards=num_shards, **kwargs)

        sentences = [
            "to keep your balance, you must keep moving",  # Albert Einstein
            "be yourself, everyone else is already taken",  # Oscar Wilde
            "the future belongs to those who believe in the beauty of their dreams",  # Eleanor Roosevelt
            "you must be the change you with to see in the world",  # Mahatma Gandhi
            "the most certain way to succeed is always to try just one more time",  # Thomas Edison
        ]
        maxlen = max(len(sentence) for sentence in sentences)
        sentences = [sentence.ljust(maxlen) for sentence in sentences]
        sentences = [
            [char_to_bitword(c, DataSimpleQuotes.VALID_CHARACTERS, 5) for c in sentence]
            for sentence in sentences
        ]

        self._batches_data = self._sentences_to_batches(sentences)

    @property
    def _batches(self) -> List[Batch]:
        return self._batches_data

    @property
    def input_width(self) -> tensor:
        return 5

    def to_human(self, target: tensor, offset: int = 0) -> str:
        return " " * offset + "".join(
            [DataSimpleQuotes.DISPLAY_CHARACTERS[bitword_to_int(c)] for c in target]
        )
