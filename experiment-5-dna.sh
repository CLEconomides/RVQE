#!/usr/bin/env bash
# mnist 8 digits

seeds=( 29885 27489 31509 25388 27175 32030 31615 30680 34899 25969 32780 30084 33470 26845 32630 28785 30883 26159 30762 34317 26305 33016 29421 25127 33282 33391 34143 31087 30698 27968 )
lengths=( 5 10 15 20 25 30 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"


trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


PORT=27777


for len in "${lengths[@]}"; do
for sd in "${seeds[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++))

    TAG="pool-$len-$sd"

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
        --port $PORT \
        --num-shards 2 \
        --epochs 1000 \
        --stop-at-loss 0.0005 \
        train \
        --dataset dna \
        --sentence-length $len \
        --workspace 6 \
        --stages 2 \
        --order 2 \
        --degree 3 \
        --optimizer adam \
        --learning-rate 0.005 \
        --batch-size 128
    
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
done
