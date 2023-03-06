import numpy as np

from RVQE.data import *


def constant_sentence(length: int, constant: Bitword) -> List[Bitword]:
    return [constant for _ in range(length)]


def alternating_sentence(length: int, constants: List[Bitword]) -> List[Bitword]:
    return [constants[i % len(constants)] for i in range(length)]


def int_to_bitstring_vector(integ, nn):
    bitstring = bin(integ)[2:].rjust(nn, "0")
    bitstring_vector = [int(letter) for letter in bitstring]
    return bitstring_vector
def min_max_normalisation(input_seq,nn):
    """renormalize the different training sequences with the min-max method, but not
    into an interval of [0,1] but into [0,2^n] of integers, so that they can be
    represented as bitstrings"""

    input_seq = np.array(input_seq)

    max_val = np.max(input_seq)
    min_val = np.min(input_seq)
    return [[int_to_bitstring_vector(int(round(((i-min_val)*(2**nn-1)/(max_val-min_val)),0)),nn) for i in sequ] for sequ in input_seq]
    # return [int_to_bitstring_vector(
    #     int(round(((i - min_val) * (2 ** nn - 1) / (max_val - min_val)), 0)), nn) for i
    #          in input_seq]

# print([[i + j**2 for i in range(j,j+3)] for j in range(2)])
# print(min_max_normalisation([[i + j**2 for i in range(3)] for j in range(2)],3))
# print(min_max_normalisation([[1,2,3],[4,5,6],[7,8,8]],3))
# print(min_max_normalisation([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.8]],3))

class QC_Finance_eval(DataFactory):
    def __init__(self, input_qubits: int, *args, **kwargs):
        self.input_qubits = input_qubits
        super().__init__(*args, **kwargs)

        #import data from csv file?
        self.sentence_length = 8
        input = [[i + 1 for i in range(self.sentence_length)]]

        self.input = input

        sentences = min_max_normalisation(input, self.input_qubits)

        self._batches_data = self._sentences_to_batches(sentences, targets=sentences)

    @property
    def _batches(self) -> List[Batch]:
        return self._batches_data

    @property
    def input_width(self) -> int:
        return self.input_qubits

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        # for word in target:
        #     print('word', word)
        return "  " * offset + " ".join([str(bitword_to_int(word)) for word in target])

class QC_Finance_train(DataFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #also import data from csv file?

        input_seq = [[i for i in range(self.sentence_length)] for j in range(self.batch_size)]

        self.input = input_seq

        sentences = min_max_normalisation(input_seq, 3)

        self._batches_data = self._sentences_to_batches(sentences, targets=sentences)

    @property
    def _batches(self) -> List[Batch]:
        return self._batches_data

    @property
    def input_width(self) -> int:
        return 3

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return "  " * offset + " ".join([str(bitword_to_int(word)) for word in target])


class DataSimpleSequences(DataFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sentences = [
            alternating_sentence(
                self.sentence_length, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
            ),
            constant_sentence(self.sentence_length, [1, 0, 0]),
        ]

        self._batches_data = self._sentences_to_batches(sentences, targets=sentences)

    @property
    def _batches(self) -> List[Batch]:
        return self._batches_data

    @property
    def input_width(self) -> int:
        return 3

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return "  " * offset + " ".join([str(bitword_to_int(word)) for word in target])


#print(DataSimpleSequences._batches)
class DataSimpleQuotes(DataFactory):
    """
        Larger memoization task; we give advice by postselecting on consonants
        For the quotes given, we have 149 consonants, and 315 characters to be predicted,
        so we give roughly 47% advice.
    """

    VALID_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! \n"
    DISPLAY_CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?! ¶"
    assert len(VALID_CHARACTERS) <= 32, "characters should fit into 5 bits"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sentences = [
            "to keep your balance, you must keep moving.",  # Albert Einstein
            "be yourself, everyone else is already taken.",  # Oscar Wilde
            "the future belongs to those who believe in the beauty of their dreams.",  # Eleanor Roosevelt
            "you must be the change you with to see in the world.",  # Mahatma Gandhi
            "the most certain way to succeed is always to try just one more time.",  # Thomas Edison
            "wir muessen wissen, wir werden wissen.",  # David Hilbert
        ]
        maxlen = max(len(sentence) for sentence in sentences)
        sentences = [sentence.ljust(maxlen) for sentence in sentences]
        sentences = [
            [char_to_bitword(c, DataSimpleQuotes.VALID_CHARACTERS, 5) for c in sentence]
            for sentence in sentences
        ]

        self._batches_data = self._sentences_to_batches(sentences, targets=sentences)

    @property
    def _batches(self) -> List[Batch]:
        return self._batches_data

    @property
    def input_width(self) -> int:
        return 5

    def to_human(self, target: torch.LongTensor, offset: int = 0) -> str:
        return " " * offset + "".join(
            [DataSimpleQuotes.DISPLAY_CHARACTERS[bitword_to_int(c)] for c in target]
        )

    CONSONANTS = "bcdfghjklmnpqrstvwxyz"

    def _ignore_output_at_step(self, index: int, target: Union[tensor, Bitword]) -> bool:
        """
            return True for consonant targets
        """
        return (
            bitword_to_char(target, DataSimpleQuotes.VALID_CHARACTERS)
            in DataSimpleQuotes.CONSONANTS
        )
