#!/usr/bin/env bash

optimizers=( "rmsprop" )
learningrates=( 0.02 0.01 0.005 0.002 )
seeds=( 1523 2342 1231 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s

for lr in "${learningrates[@]}"
do
    for sd in "${seeds[@]}"
    do
        for optim in "${optimizers[@]}"
        do
            LOCKFILE="$LOCKFILEFOLDER/experiment2-$sd-$optim-$lr.lock"
            DONEFILE="$LOCKFILEFOLDER/experiment2-$sd-$optim-$lr.done"
            sync

            if [[ ! -f "$DONEFILE" ]] ; then
                {
                    if flock -n 200 ; then
                        echo "running $optim with $lr"
                        ./main.py \
                            --tag experiment2-$sd-$optim-$lr \
                            --seed $sd \
                            --num-shards 60 \
                            --epochs 2500 \
                            train \
                            --dataset elman-xor \
                            --workspace 6 \
                            --stages 6 \
                            --order 2 \
                            --degree 2 \
                            --optimizer $optim \
                            --learning-rate $lr \
                            --sentence-length 12 \
                            --batch-size 2
                        
                        if  [[ $? -eq 0 ]] ; then
                            touch "$DONEFILE"                
                            sync
                        else
                            echo "failure running $optim with $lr."
                        fi
                        sleep 1

                    else
                        echo "skipping $optim with $lr"
                    fi
                } 200>"$LOCKFILE"
            fi
        done
    done
done