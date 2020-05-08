#!/usr/bin/env bash

orders=( 1 2 3 )
degrees=( 1 2 3 )
stages=( 1 2 3 4 5 )
workspaces=( 1 2 3 4 5 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s

for o in "${orders[@]}"
do
    for d in "${degrees[@]}"
    do
        for s in "${stages[@]}"
        do
            for w in "${workspaces[@]}"
            do

                LOCKFILE="$LOCKFILEFOLDER/experiment3-$o-$d-$s-$w.lock"
                DONEFILE="$LOCKFILEFOLDER/experiment3-$o-$d-$s-$w.done"
                sync

                if [[ ! -f "$DONEFILE" ]] ; then
                    {
                        if flock -n 200 ; then
                            echo "running $o-$d-$s-$w"
                            ./main.py \
                                --tag experiment3-$o-$d-$s-$w \
                                --seed 2349711 \
                                --num-shards 2 \
                                --epochs 500 \
                                train \
                                --dataset simple-seq \
                                --workspace $w \
                                --stages $s \
                                --order $o \
                                --degree $d \
                                --optimizer rmsprop \
                                --learning-rate 0.01 \
                                --sentence-length 20 \
                                --batch-size 1
                            
                            if  [[ $? -eq 0 ]] ; then
                                touch "$DONEFILE"                
                                sync
                            else
                                echo "failure running $o-$d-$s-$w."
                            fi
                            sleep 1

                        else
                            echo "skipping $o-$d-$s-$w"
                        fi
                    } 200>"$LOCKFILE"
                fi
            done
        done
    done
done