"""
    Elman's paper "Finding Structure in Time",
    https://crl.ucsd.edu/~elman/Papers/fsit.pdf
"""


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
        return 1


    def next_batch(self) -> Batch:
        # extract random batch of xor sequences like 011 101 110 000 ...
        sentences = []
        targets = []
        while len(sentences) < self.batch_size:
            sentence = []
            target = []
            for _ in range(0, self.sentence_length, 3):
                a, b = torch.randint( 0, 2, (2,), generator=self.rng ).tolist()
                c = a ^ b
                sentence += [[a], [b], [c]]
                target += [0, 0, c]
            sentence = sentence[:self.sentence_length]
            target = target[:self.sentence_length]

            sentences.append(sentence)
            targets.append(target)

        # turn into batch
        return tensor(sentences), tensor(targets)
        
    def to_human(self, target: tensor, offset: int = 0) -> str:
        if offset == 0:  # gold
            return " ".join([
                "".join([ str(x[0]) for x in triple.tolist()]) for triple in torch.split(target, 3)
            ])
        elif offset == 1:  # comparison

            small = lambda n: '₀' if n == 0 else '₁'
            
            out = f" {small(target[0])}{target[1].item()}"
            for triple in torch.split(target[2:], 3):
                out += " " + small(triple[0])
                if len(triple) > 1:
                    out += small(triple[1])
                if len(triple) > 2:
                    out += f"{triple[2].item()}"
            return out



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
    LETTER_LUT = {
        "b": "ba",
        "d": "dii",
        "g": "guuu"
    }

    BITWORD_LUT = {
        "b": [1, 0, 1, 0, 0, 1],
        "d": [1, 0, 1, 1, 0, 1],
        "g": [1, 0, 1, 0, 1, 1],
        "a": [0, 1, 0, 0, 1, 1],
        "i": [0, 1, 0, 1, 0, 1],
        "u": [0, 1, 0, 1, 1, 1]
    }

    TARGET_LUT = {
        "b": 0,  # marker for arbitrary consonant
        "d": 0,
        "g": 0,
        "a": bitword_to_int(BITWORD_LUT["a"]),
        "i": bitword_to_int(BITWORD_LUT["i"]),
        "u": bitword_to_int(BITWORD_LUT["u"])
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
            bdg_seq = torch.randint( 0, 3, (self.sentence_length,), generator=self.rng ).tolist()
            bdg_seq = [ ("b", "d", "g")[i] for i in bdg_seq ]

            # replace with words
            bdg_aiu_seq = "".join([ self.LETTER_LUT[c] for c in bdg_seq ])[: self.sentence_length]

            # replace with vectors
            sentence = [ self.BITWORD_LUT[c] for c in bdg_aiu_seq ]
            target = [ self.TARGET_LUT[c] for c in bdg_aiu_seq ]

            sentences.append(sentence)
            targets.append(target)

        # turn into batch
        return tensor(sentences), tensor(targets)

    INVERSE_TARGET_LUT = {
        41: " b",
        45: " d",
        43: " g",
        19: "a",
        21: "i",
        23: "u",
         0: " ₀"  # extra marker for target when we expect a consonant
    }
    
    def to_human(self, target: tensor, offset: int = 0) -> str:
        target = [ bitword_to_int(t) for t in target ]
        # start with offset number of blanks
        out = " "*offset

        # append string
        out += "".join([ self.INVERSE_TARGET_LUT[c] if c in self.INVERSE_TARGET_LUT else "?" for c in target ])

        # if we start with a consonant, trim one space off
        return out if target[0] not in [41, 45, 43, 0] else out[1:]

