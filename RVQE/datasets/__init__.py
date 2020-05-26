from .shakespeare import DataShakespeare
from .simple import DataSimpleSequences, DataSimpleQuotes
from .elman import DataElmanXOR, DataElmanLetter
from .mnist import DataMNIST01, DataMNIST36, DataMNIST01_Gen, DataMNIST

all_datasets = {
    "simple-seq": DataSimpleSequences,
    "simple-quotes": DataSimpleQuotes,
    "elman-xor": DataElmanXOR,
    "elman-letter": DataElmanLetter,
    "mnist01": DataMNIST01,
    "mnist36": DataMNIST36,
    "mnist01-gen": DataMNIST01_Gen,
    "mnist": DataMNIST,
    "shakespeare": DataShakespeare,
}
