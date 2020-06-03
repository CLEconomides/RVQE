"""
    DNA Sequences
"""

import colorful

from ..data import *


class DataDNA(DataFactory):
    """
        we map GATC to 2-bit-strings; and U to the value 4.
        The task is to identify the DNA base following the single "C"
        which appears somewhere within the sequence
    """

    def __init__(self, shard: int, **kwargs):
        super().__init__(shard, **kwargs)

    @property
    def _batches(self) -> List[Batch]:
        raise NotImplementedError("next_batch overridden")

    @property
    def input_width(self) -> int:
        return 3

    def next_batch(self, _, __) -> Batch:
        # extract random batch of xor sequences like 011 101 110 000 ...
        sentences = []
        targets = []
        while len(sentences) < self.batch_size:
            sentence = torch.randint(0, 4, (self.sentence_length,), generator=self.rng).tolist()
            target = torch.zeros((self.sentence_length,)).int().tolist()

            # position of "C" anywhere but at the first and last index
            idx_C = torch.randint(1, self.sentence_length - 1, (1,), generator=self.rng).item()
            sentence[idx_C] = 4
            target[idx_C] = 4

            # last target item has to be base that follows "A"
            target[-1] = sentence[idx_C + 1]

            # binarize
            sentence = [int_to_bitword(s, 3) for s in sentence]
            target = [int_to_bitword(t, 3) for t in target]

            sentences.append(sentence)
            targets.append(target)

        # turn into batch
        return self._sentences_to_batch(sentences, targets)

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        target = tensor([bitword_to_int(t) for t in target])

        if offset == 0:  # gold
            # heuristic to differentiate between input and target
            is_input = sum(target) > 7
            idx_C = (target == 4).nonzero()[0, 0].item()
            base_following_C = "GATCU"[target[idx_C + 1 if is_input else -1].item()]

            return f"base sequence U{colorful.bold(base_following_C)}  @  ago {self.sentence_length - idx_C}"

        elif offset == 1:  # comparison
            base_following_C = "GATCU???"[target[-1].item()]

            return f"base sequence U{colorful.bold(base_following_C)}"

    def filter(
        self,
        sequence: torch.LongTensor,
        *,
        dim_sequence: int,
        targets_hint: torch.LongTensor,
        dim_targets: int,
    ) -> torch.LongTensor:
        """
            we only care about the item under idx_C, and the last one
        """
        assert sequence.dim() == 3 and dim_sequence in [1, 2]
        assert targets_hint.dim() == 3 and dim_targets in [1, 2]

        targets_hint = targets_hint if dim_targets == 1 else targets_hint.transpose(1, 2)
        assert targets_hint.shape[-1] == self.input_width
        sequence = sequence if dim_sequence == 1 else sequence.transpose(1, 2)
        assert sequence.shape[:2] == targets_hint.shape[:2]

        # prepare new output
        sequence_out = []
        for s, t in zip(sequence, targets_hint):
            idx_C = (t == tensor([1, 0, 0])).all(dim=1).nonzero()[0, 0].item()
            sequence_out.append(s[[idx_C, -1]])
        sequence_out = torch.stack(sequence_out)

        # restore old shape
        return sequence_out if dim_sequence == 1 else sequence_out.transpose(1, 2)

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            we just have to find the idx_C or the last
            we expect target to have length self.sequence_length - 1, so the last index
            is self.sequence_length - 2
        """
        return (index != self.sentence_length - 2) and not (target == tensor([1, 0, 0])).all()
