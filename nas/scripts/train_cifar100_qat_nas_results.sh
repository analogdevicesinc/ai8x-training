#!/bin/sh
MODEL_NAME_PREFIX="ai85nasnet_cifar100_res"
NUM_MODELS=10

./nas_auto_generator.py -i nas/nas_out_subnets_cifar100.json -n $MODEL_NAME_PREFIX

for i in $(seq 1 $NUM_MODELS)
do  
   echo "Training model ${i}"
   ./train.py --deterministic --epochs 300 --optimizer Adam --lr 0.001 --compress schedule-cifar-nas.yaml --model "${MODEL_NAME_PREFIX}_${i}" --dataset CIFAR100 --device MAX78000 --batch-size 100 --print-freq 100 --validation-split 0 --use-bias --qat-policy qat_policy_late_cifar.yaml "$@"
done
