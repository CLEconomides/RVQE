#!/usr/bin/env bash

optimizers=( "sgd" "rmsprop" "adam" )
learningrates=( 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 )
seeds=( 1523 2342 1231 9948 2349 5675 4358 9389 9999 2525 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

sleep $[ ($RANDOM % 400) + 1 ]s

for sd in "${seeds[@]}"
do
    for lr in "${learningrates[@]}"
    do
        for optim in "${optimizers[@]}"
        do
            LOCKFILE="$LOCKFILEFOLDER/experiment2-$seed-$optim-$lr.lock"
            DONEFILE="$LOCKFILEFOLDER/experiment2-$seed-$optim-$lr.done"
            sync
            if [[ ! -f "$LOCKFILE" && ! -f "$DONEFILE" ]]
            then
                touch "$LOCKFILE"
                sync
                echo "running $optim with $lr"
                ./main.py --tag experiment2-$seed-$optim-$lr --seed $sd --num-shards 3 --epochs 500 train --dataset elman-xor --stages 3 --optimizer $optim --learning-rate $lr --sentence-length 3 --batch-size 4
                touch "$DONEFILE"
                sync
                sleep 1
                rm "$LOCKFILE"
            else
                echo "skipping $optim with $lr and seed $seed"
            fi
        done
    done
done