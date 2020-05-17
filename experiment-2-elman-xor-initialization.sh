#!/usr/bin/env bash
# elman-letter

biases=( 0 .785398 1.5708 3.14159 )
spreadsBias=( 0 0.01 0.1 1. 10. )
spreadsWeights=( 0 0.01 0.1 1. 10. )
spreadsUnitaries=( 0 0.01 0.1 1. 10. )
seeds=( 6936 5081 3774 7427 700 1664 7262 499 9736 6654 )

LOCKFILEFOLDER="./locks"
mkdir -p "$LOCKFILEFOLDER"

trap "exit" INT
sleep $[ ($RANDOM % 40) + 1 ]s


for sd in "${seeds[@]}"
do
for bias in "${biases[@]}"
do
for spreadBias in "${spreadsBias[@]}"
do
for spreadWeight in "${spreadsWeights[@]}"
do
for spreadUnitary in "${spreadsUnitaries[@]}"
do
    TAG="elman-xor-initialization-$sd-$bias-$spreadBias-$spreadWeight-$spreadUnitary"

    LOCKFILE="$LOCKFILEFOLDER/experiment-$TAG.lock"
    DONEFILE="$LOCKFILEFOLDER/experiment-$TAG.done"
    FAILFILE="$LOCKFILEFOLDER/experiment-$TAG.fail"
    sync

    # check if any lockfiles present
    if [[ -f "$DONEFILE" || -f "$FAILFILE" || -f "$LOCKFILE" ]] ; then
        echo "skipping $TAG"
        break
    fi

    # try to aquire lockfile
    exec 200>"$LOCKFILE"
    flock -n 200 || {
        echo "skipping $TAG"
        break
    }
    
    # run test
    echo "running $TAG"
    ./main.py \
        --tag experiment-$TAG \
        --seed $sd \
        --num-shards 2 \
        --epochs 1000 \
        --stop-at-loss 0.001 \
        train \
        --dataset elman-xor \
        --workspace 4 \
        --stages 1 \
        --order 2 \
        --degree 3 \
        --optimizer adam \
        --learning-rate 0.005 \
        --sentence-length 36 \
        --batch-size 8 \
        --initial-bias $bias \
        --initial-bias-spread $spreadBias \
        --initial-weights-spread $spreadWeight \
        --initial-unitaries-spread $spreadUnitary
    
    if  [[ $? -eq 0 ]] ; then
        touch "$DONEFILE"                
        sync
    else
        touch "$FAILFILE"            
        sync      
        echo "failure running $TAG."
    fi
    sleep 1

done
done
done
done
done