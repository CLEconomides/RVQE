#!/usr/bin/env bash

optimizers=( "sgd" "rmsprop" "adam" )
learningrates=( 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 )

for optim in "${optimizers[@]}"
do
    for lr in "${learningrates[@]}"
    do
        ./main.py --tag lr-optim-test --epochs 500 train --optimizer $optim --learning-rate $lr
    done
done