"""
    Elman's paper "Finding Structure in Time",
    https://crl.ucsd.edu/~elman/Papers/fsit.pdf
"""

import colorful

from ..data import *


class DataElmanXOR(DataFactory):
    """
        Elman's paper on XOR sequence prediction.
        
        Given an input like 000 011 110 110 101 ...
        where the bits follow the rule third = first ^ second,
        provided as continuous sequence, the network should, character-by-character,
        predict the next digit.

        Naturally, only the _third_ bit can be predicted; the first and second cannot.
        This makes the training somewhat harder.
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)

        # local rng
        self.rng = torch.Generator().manual_seed(8742 + shard)

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> tensor:
        return 2

    def next_batch(self) -> Batch:
        # extract random batch of xor sequences like 011 101 110 000 ...
        sentences = []
        targets = []
        while len(sentences) < self.batch_size:
            sentence = []
            target = []
            for _ in range(0, self.sentence_length, 3):
                a, b = torch.randint(0, 2, (2,), generator=self.rng).tolist()
                c = a ^ b
                sentence += [[0, a], [0, b], [0, c]]
                target += [[1, 0], [1, 0], [0, c]]
            sentence = sentence[: self.sentence_length]
            target = target[: self.sentence_length]

            sentences.append(sentence)
            targets.append(target)

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    def to_human(self, target: tensor, offset: int = 0) -> str:
        def to_str(item: tensor) -> str:
            item = bitword_to_int(item)
            return str(item) if item != 2 else "·"

        def style_triple(triple: List[int]) -> str:
            if len(triple) > 0:
                out = to_str(triple[0])
            if len(triple) > 1:
                out += to_str(triple[1])
            if len(triple) > 2:
                out += colorful.bold(to_str(triple[2]))
            return out

        if offset == 0:  # gold
            return " ".join([style_triple(triple.tolist()) for triple in torch.split(target, 3)])
        elif offset == 1:  # comparison
            out = " " + to_str(target[0]) + colorful.bold(to_str(target[1])) + " "
            return out + " ".join(
                [style_triple(triple.tolist()) for triple in torch.split(target[2:], 3)]
            )

    def filter(self, sequence: tensor, dim: int) -> tensor:
        """
            we expect these to be offset by 1 from a proper output, i.e.
            01 110 000 011
             |   |   |   |
            and skip all elements other than that in the given direction
        """
        assert sequence.dim() == 3 and dim in [1, 2]

        if dim == 1:
            return sequence[:, 1::3, :]
        elif dim == 2:
            return sequence[:, :, 1::3]

    def filter_sentence(self, sentence: tensor) -> tensor:
        return sentence[1::3]

    def ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            return True for the steps
            01 110 000 011
            |  ||  ||  || 
        """
        return (index + 2) % 3 != 0


class DataElmanLetter(DataFactory):
    """
        Elman's paper on letter sequence prediction.
        
        We produce a random sequence of the consonants b, d, g;
        then perform the replacements 

            b -> ba
            d -> dii
            g -> guuu

        and map it to a six-bit word via the table given below.
        
        Not all letters can be predicted; but given a consonant,
        the following letters should be predictable.
    """

    LETTER_LUT = {"b": "ba", "d": "dii", "g": "guuu"}

    BITWORD_LUT = {
        "b": [1, 0, 1, 0, 0, 1],
        "d": [1, 0, 1, 1, 0, 1],
        "g": [1, 0, 1, 0, 1, 1],
        "a": [0, 1, 0, 0, 1, 1],
        "i": [0, 1, 0, 1, 0, 1],
        "u": [0, 1, 0, 1, 1, 1],
    }

    TARGET_LUT = {
        "b": [0] * 6,  # marker for arbitrary consonant
        "d": [0] * 6,
        "g": [0] * 6,
        "a": BITWORD_LUT["a"],
        "i": BITWORD_LUT["i"],
        "u": BITWORD_LUT["u"],
    }

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)

        # local rng
        self.rng = torch.Generator().manual_seed(8742 + shard)

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> tensor:
        return 6

    def next_batch(self) -> Batch:
        sentences = []
        targets = []
        while len(sentences) < self.batch_size:
            # create random sequence of b, d, g
            bdg_seq = torch.randint(0, 3, (self.sentence_length,), generator=self.rng).tolist()
            bdg_seq = [("b", "d", "g")[i] for i in bdg_seq]

            # replace with words
            bdg_aiu_seq = "".join([self.LETTER_LUT[c] for c in bdg_seq])[: self.sentence_length]

            # replace with vectors
            sentence = [self.BITWORD_LUT[c] for c in bdg_aiu_seq]
            target = [self.TARGET_LUT[c] for c in bdg_aiu_seq]

            sentences.append(sentence)
            targets.append(target)

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    INVERSE_TARGET_LUT = {
        41: " b",
        45: " d",
        43: " g",
        19: "a",
        21: "i",
        23: "u",
        0: " ·",  # extra marker for target when we expect a consonant
    }

    def to_human(self, target: tensor, offset: int = 0) -> str:
        target = [bitword_to_int(t) for t in target]
        # start with offset number of blanks
        out = " " * offset

        # append string
        out += "".join(
            [self.INVERSE_TARGET_LUT[c] if c in self.INVERSE_TARGET_LUT else "?" for c in target]
        )

        # if we start with a consonant, trim one space off
        return out if target[0] not in [41, 45, 43, 0] else out[1:]

    def ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            return True for consonant targets
        """
        return bitword_to_int(target) in [41, 45, 43]
