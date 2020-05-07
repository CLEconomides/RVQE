#!/usr/bin/env bash

optimizers=( "sgd" "rmsprop" "adam" "lbfgs" )
learningrates=( 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 10) + 1 ]s

for optim in "${optimizers[@]}"
do
    for lr in "${learningrates[@]}"
    do
        LOCKFILE="$LOCKFILEFOLDER/experiment1-$optim-$lr.lock"
        DONEFILE="$LOCKFILEFOLDER/experiment1-$optim-$lr.done"
        sync
        
        if [[ ! -f "$DONEFILE" ]]
        then
            # acquire lock
            if ( set -o noclobber; echo "locked" > "$LOCKFILE") 2> /dev/null; then
                trap 'rm -f "$LOCKFILE"; exit $?' INT TERM EXIT

                echo "running $optim with $lr"
                ./main.py --tag experiment1-$optim-$lr --epochs 500 train --optimizer $optim --learning-rate $lr
                touch "$DONEFILE"
                sync
                sleep 1

                trap "exit" INT
                rm -f "$LOCKFILE"
            else
                echo "skipping $optim with $lr"
            fi
        fi
    done
done