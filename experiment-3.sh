#!/usr/bin/env bash

orders=( 1 2 3 )
stages=( 1 2 3 4 5 6 7 8 )
workspaces=( 2 3 4 5 6 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

sleep $[ ($RANDOM % 400) + 1 ]s

for o in "${orders[@]}"
do
    for s in "${stages[@]}"
    do
        for w in "${workspaces[@]}"
        do
            LOCKFILE="$LOCKFILEFOLDER/experiment3-$o-$s-$w-.lock"
            DONEFILE="$LOCKFILEFOLDER/experiment3-$o-$s-$w-.done"
            sync
            if [[ ! -f "$LOCKFILE" && ! -f "$DONEFILE" ]]
            then
                touch "$LOCKFILE"
                sync
                echo "running order $o, stages $s, workspace $w"
                (( port = 16337 + 100 * $w + 10 * $s + $o ))
                ./main.py --port $port --num-shards 2 --tag experiment3-$o-$s-$w --epochs 500 train --dataset elman-xor --stages $s --workspace $w --order $o --optimizer rmsprop --learning-rate 0.15 --batch-size 1
                touch "$DONEFILE"
                sync
                sleep 1
                rm "$LOCKFILE"
            else
                echo "skipping order $o, stages $s, workspace $w"
            fi
        done
    done
done