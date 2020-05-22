#!/usr/bin/env bash
# mnist01

seeds=( 51801 19342 35675 28847 28722 38358 75343 62339 71350 47611 64490 51422 63763 17796 35557 83224 40797 47253 87583 23424 83502 93806 52495 54785 74155 65123 69520 78119 46386 76720 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


for sd in "${seeds[@]}"; do

    TAG="mnist01-$sd"

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
        --num-shards 3 \
        --epochs 5000 \
        train \
        --dataset mnist01-gen \
        --workspace 8 \
        --stages 2 \
        --order 2 \
        --degree 2 \
        --optimizer adam \
        --learning-rate 0.005 \
        --batch-size 16
    
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
