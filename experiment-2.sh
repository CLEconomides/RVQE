#!/usr/bin/env bash

optimizers=( "rmsprop" )
learningrates=( 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 )
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
                            --num-shards 10 \
                            --epochs 2500 \
                            train \
                            --dataset elman-xor \
                            --workspace 10 \
                            --stages 5 \
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


            if [[ ! -f "$LOCKFILE" && ! -f "$DONEFILE" ]]
            then
                touch "$LOCKFILE"
                sync
                echo "running $optim with $lr"
                ./main.py --tag experiment2-$sd-$optim-$lr --seed $sd --num-shards 3 --epochs 500 train --dataset elman-xor --stages 3 --optimizer $optim --learning-rate $lr --sentence-length 3 --batch-size 4
                status=$?
                
                if test $status -eq 0
                then
                    touch "$DONEFILE"
                    sync
                    sleep 1
                    rm "$LOCKFILE"
                else
                    echo "failure running $optim with $lr."
                    rm "$LOCKFILE"
                fi                
            else
                echo "skipping $optim with $lr and seed $sd"
            fi
        done
    done
done