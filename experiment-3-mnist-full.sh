#!/usr/bin/env bash
# mnist01

seeds=( 29203 39060 35605 88615 30218 62354 22195 23481 14629 58825 78763 74317 54341 59416 12173 57478 51841 36516 68534 69542 86216 50816 21675 77289 33313 13904 68515 49315 61697 89319 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


for sd in "${seeds[@]}"; do

    TAG="mnist-full-$sd"

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
        --dataset mnist \
        --workspace 8 \
        --stages 1 \
        --order 2 \
        --degree 2 \
        --optimizer adam \
        --learning-rate 0.005 \
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
