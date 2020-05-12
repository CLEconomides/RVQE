#!/usr/bin/env bash
# test which optimizer works within which regime
# this experiment is very neat and simple; keep

optimizers=( "sgd" "rmsprop" "adam" )
learningrates=( 10.0 5.0 2.0 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 10) + 1 ]s

for optim in "${optimizers[@]}"
do
    for lr in "${learningrates[@]}"
    do
        TAG="optimizer-$optim-$lr"

        LOCKFILE="$LOCKFILEFOLDER/experiment-$TAG.lock"
        DONEFILE="$LOCKFILEFOLDER/experiment-$TAG.done"
        sync
        
        if [[ ! -f "$DONEFILE" ]] ; then
            {
                if flock -n 200 ; then
                    echo "running $TAG"
                    ./main.py --tag experiment-$TAG --epochs 500 train --optimizer $optim --learning-rate $lr --workspace 5 --stages 5 --order 2
                    touch "$DONEFILE"
                    sync
                    sleep 1
                else
                    echo "skipping $TAG"
                fi
            } 200>"$LOCKFILE"
        fi
    done
done
