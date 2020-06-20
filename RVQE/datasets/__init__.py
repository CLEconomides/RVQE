from .shakespeare import DataShakespeare
from .simple import DataSimpleSequences, DataSimpleQuotes
from .elman import DataElmanXOR, DataElmanLetter
from .mnist import *
from .mnistpca import *
from .mnisttsne import *
from .dna import *

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
    "mnist01-ds-lrg": DataMNIST01ds_lrg,
    "mnist36-ds-lrg": DataMNIST36ds_lrg,
    "mnist8-ds-lrg": DataMNIST8ds_lrg,
    "mnist-pca-r2-p2": DataMNISTPCA_r2p2,
    "mnist-pca-r2-p5": DataMNISTPCA_r2p5,
    "mnist-pca-r2-p8": DataMNISTPCA_r2p8,
    "mnist-pca-r3-p2": DataMNISTPCA_r3p2,
    "mnist-pca-r3-p5": DataMNISTPCA_r3p5,
    "mnist-pca-r3-p8": DataMNISTPCA_r3p8,
    "mnist-pca-r4-p2": DataMNISTPCA_r4p2,
    "mnist-pca-r4-p5": DataMNISTPCA_r4p5,
    "mnist-pca-r4-p8": DataMNISTPCA_r4p8,
    "mnist-tsne-d2-r2": DataMNISTtSNE_d2r2,
    "mnist-tsne-d2-r3": DataMNISTtSNE_d2r3,
    "mnist-tsne-d2-r4": DataMNISTtSNE_d2r4,
    "mnist-tsne-d2-r5": DataMNISTtSNE_d2r5,
    "mnist-tsne-d2-r8": DataMNISTtSNE_d2r8,
    "mnist-tsne-d3-r2": DataMNISTtSNE_d3r2,
    "mnist-tsne-d3-r3": DataMNISTtSNE_d3r3,
    "mnist-tsne-d3-r4": DataMNISTtSNE_d3r4,
    "mnist-tsne-d3-r5": DataMNISTtSNE_d3r5,
    "mnist-tsne-d3-r8": DataMNISTtSNE_d3r8,
    "mnist01-gen": DataMNIST01_Gen,
    "shakespeare": DataShakespeare,
    "dna": DataDNA,
}
