#!/usr/bin/env bash
# mnist 8 digits

seeds=( 1000 1010 1020 1030 1040 1050 1060 1070 )
datasets=( "mnist-tsne-d2-r4" "mnist-tsne-d3-r3" )
lrs=( 0.03 0.01 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"


trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


PORT=37777


for sd in "${seeds[@]}"; do
for dataset in "${datasets[@]}"; do
for lr in "${lrs[@]}"; do
    # increment port in case multiple runs on same machine
    ((PORT++))

    TAG="pool-$dataset-$sd-$lr"

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
        --epochs 10000 \
        train \
        --dataset $dataset \
        --workspace 7 \
        --stages 2 \
        --order 2 \
        --degree 2 \
        --optimizer adam \
        --learning-rate $lr \
        --batch-size 32
    
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
done
