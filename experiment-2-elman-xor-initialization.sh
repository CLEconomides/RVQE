#!/usr/bin/env bash
# elman-letter

biases=( 0 .785398 1.5708 4.71239 )
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
                    --dataset elman-xor \
                    --workspace 6 \
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
done
done