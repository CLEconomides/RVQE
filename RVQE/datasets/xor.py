
from ..data import *

class DataXOR(DataFactory):
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
