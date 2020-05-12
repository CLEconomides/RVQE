#!/usr/bin/env bash
# elman-xor

wds=( 0.1 0.01 0.001 0.0001 0 )
seeds=( 720 7292 4402 5427 4269 7928 3475 5114 3975 2733 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s

for sd in "${seeds[@]}"
do
    for wd in "${wds[@]}"
    do
        TAG="elman-xor-$sd-$wd"

        LOCKFILE="$LOCKFILEFOLDER/experiment-$TAG.lock"
        DONEFILE="$LOCKFILEFOLDER/experiment-$TAG.done"
        sync

        if [[ ! -f "$DONEFILE" ]] ; then
            {
                if flock -n 200 ; then
                    echo "running $TAG"
                    ./main.py \
                        --tag experiment-$TAG \
                        --seed $sd \
                        --num-shards 3 \
                        --epochs 1000 \
                        train \
                        --dataset elman-xor \
                        --workspace 6 \
                        --stages 3 \
                        --order 2 \
                        --degree 4 \
                        --optimizer rmsprop \
                        --learning-rate 0.01 \
                        --weight-decay $wd \
                        --sentence-length 36 \
                        --batch-size 8
                    
                    if  [[ $? -eq 0 ]] ; then
                        touch "$DONEFILE"                
                        sync
                    else
                        echo "failure running $TAG."
                    fi
                    sleep 1

                else
                    echo "skipping $TAG"
                fi
            } 200>"$LOCKFILE"
        fi
    done
done