#!/usr/bin/env bash
# mnist01 or mnist36

seeds=( 48225 52473 38637 88743 17613 19587 17415 15629 77153 97512 83937 37571 24434 12186 21750 66718 37375 54635 24830 88130 95737 50215 58711 69903 23407 44170 74966 33622 34931 13741 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

DATASET=$1

if [[ "$DATASET" != "mnist01" && "$DATASET" != "mnist01-ds" && "$DATASET" != "mnist36" && "$DATASET" != "mnist36-ds" && "$DATASET" != "mnist01-gen" ]] ; then
    echo "invalid dataset $DATASET"
    exit 1
fi

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


for sd in "${seeds[@]}"; do

    TAG="$DATASET-$sd"

    LOCKFILE="$LOCKFILEFOLDER/experiment-$TAG.lock"
    DONEFILE="$LOCKFILEFOLDER/experiment-$TAG.done"
    FAILFILE="$LOCKFILEFOLDER/experiment-$TAG.fail"

    # check if any lockfiles present
    sync
    if [[ -f "$DONEFILE" || -f "$FAILFILE" || -f "$LOCKFILE" ]] ; then
        echo "skipping $TAG"
        continue
    fi

    # try to aquire lockfile
    exec 200>"$LOCKFILE"
    flock -n 200 || {
        echo "skipping $TAG"
        continue
    }
    
    # run test
    echo "running $TAG"
    ./main.py \
        --tag experiment-$TAG \
        --seed $sd \
        --num-shards 2 \
        --epochs 5000 \
        train \
        --dataset $DATASET \
        --workspace 8 \
        --stages 2 \
        --order 2 \
        --degree 2 \
        --optimizer adam \
        --learning-rate 0.005 \
        --batch-size 50
    
    if  [[ $? -eq 0 ]] ; then
        touch "$DONEFILE"    
    else
        touch "$FAILFILE"    
        echo "failure running $TAG."
    fi 

    sync   
    sleep 10
    rm "$LOCKFILE"
    sync   
    sleep 10
    
done
