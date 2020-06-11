# QRNN Pytorch Implementation

## Setup

The code requires only a rudimentary pytorch/tensorboard installation; training is (so far) only done on the CPU, so no CUDA setup is required.
Further packages required:

    pip install colorful

Optional packages:

    pip install pytest                       # for tests
    pip install jupyter matplotlib opentsne  # to run jupyter notebook for MNIST data augmentation


## Folder Structure

    ./notebooks/        # contains jupyter notebook for MNIST t-SNE augmentation,
                        # model evaluations, and RNN/LSTM reference implementation for DNA sequence test
    ./RVQE/             # implementation of QRNN as pytorch model
    ./RVQE/datasets/    # datasets and necessary resources

    ./main.py           # main training program

    ./*.sh              # various experiment presets used in the paper
                        # modify these to match your training environment



## Running

    ./main.py --help
    ./main.py train --help
    ./main.py resume --help

    pytest              # executes a series of tests

    ./main.py train     # executes a simple default training task (default parameters from --help)


For instance, to train the t-SNE augmented MNIST dataset on an 8-core machine, execute the following:

    OMP_NUM_THREADS=2 ./main.py \
        --tag experiment-test \
        --seed 42 \
        --port 20023 \
        --num-shards 4 \
        --epochs 10000 \
        train \
        --dataset mnist-tsne \
        --workspace 6 \
        --stages 2 \
        --order 2 \
        --degree 3 \
        --optimizer adam \
        --learning-rate 0.005 \
        --batch-size 16

Note that memory requirements go linear in the number of shards (ranks); linear in the number of stages; and grow exponentially in the workspace size. There is more parameters that control training, stopping, etc..

When training is interrupted, a checkpoint can simply be restarted with

    ./main.py resume checkpoint-name.tar.gz

A few of the parameters can be overridden, e.g. a new learning rate can be set.


## Datasets

No external datasets are necessary; the implemented ones can be elected with the `--dataset` switch. Some datasets for memorizing sequences require a certain batch size or number of shards or combinations thereof; the program will complain if this is manually set to something invalid (e.g., `--dataset simple-seq` makes the QRNN learn precisely two sequences, so the setup is such that either `--num-shards 2 ... --batch-size 1` or `--num-shards 1 ... --batch-size 2` is allowed). Most datasets do not have such a restriction.

For MNIST, the batch size indicates how many samples _of each digit_ are presented.