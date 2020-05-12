#!/usr/bin/env bash
# elman-letter

workspaces( 3 4 5 6 7 )
stages=( 1 2 3 4 )
seeds=( 6936 5081 3774 7427 700 1664 7262 499 9736 6654 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


for sd in "${seeds[@]}"
do
for ws in "${workspaces[@]}"
do
for st in "${stages[@]}"
do
    TAG="elman-letter-$sd-$ws-$st"

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
                    --num-shards 2 \
                    --epochs 1000 \
                    train \
                    --dataset elman-letter \
                    --workspace $ws \
                    --stages $st \
                    --order 2 \
                    --degree 3 \
                    --optimizer rmsprop \
                    --learning-rate 0.002 \
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
done