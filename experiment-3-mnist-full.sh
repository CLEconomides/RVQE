#!/usr/bin/env bash
# mnist 8 digits

seeds=( 23482 29328 23928 38282 24484 34382 28888 30114 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

DATASET=$1

if [[ "$DATASET" != "mnist8" && "$DATASET" != "mnist8-ds" && "$DATASET" != "mnist8-ds-lrg" ]] ; then
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
    OMP_NUM_THREADS=2 ./main.py \
        --tag experiment-$TAG \
        --seed $sd \
        --port $sd \
        --num-shards 4 \
        --epochs 5000 \
        train \
        --dataset $DATASET \
        --workspace 10 \
        --stages 2 \
        --order 2 \
        --degree 3 \
        --optimizer adam \
        --learning-rate 0.0025 \
        --batch-size 10
    
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
