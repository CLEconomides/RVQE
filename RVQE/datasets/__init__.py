from .shakespeare import DataShakespeare
from .simple import DataSimpleSequences, DataSimpleQuotes
from .elman import DataElmanXOR, DataElmanLetter
from .mnist import *

all_datasets = {
    "simple-seq": DataSimpleSequences,
    "simple-quotes": DataSimpleQuotes,
    "elman-xor": DataElmanXOR,
    "elman-letter": DataElmanLetter,
    "mnist01": DataMNIST01,
    "mnist36": DataMNIST36,
    "mnist8": DataMNIST8,
    "mnist01-ds": DataMNIST01ds,
    "mnist36-ds": DataMNIST36ds,
    "mnist8-ds": DataMNIST8ds,
    "mnist01-gen": DataMNIST01_Gen,
    "shakespeare": DataShakespeare,
}
