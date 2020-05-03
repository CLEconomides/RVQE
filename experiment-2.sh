#!/usr/bin/env bash

optimizers=( "sgd" "rmsprop" "adam" )
learningrates=( 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

sleep $[ ($RANDOM % 400) + 1 ]s

for optim in "${optimizers[@]}"
do
    for lr in "${learningrates[@]}"
    do
        LOCKFILE="$LOCKFILEFOLDER/experiment2-$optim-$lr.lock"
        DONEFILE="$LOCKFILEFOLDER/experiment2-$optim-$lr.done"
        sync
        if [[ ! -f "$LOCKFILE" && ! -f "$DONEFILE" ]]
        then
            touch "$LOCKFILE"
            sync
            echo "running $optim with $lr"
            ./main.py --tag experiment2-$optim-$lr --epochs 500 train --dataset elman-xor --stages 4 --optimizer $optim --learning-rate $lr --sentence-length 21 --batch-size 3
            touch "$DONEFILE"
            sync
            sleep 1
            rm "$LOCKFILE"
        else
            echo "skipping $optim with $lr"
        fi
    done
done