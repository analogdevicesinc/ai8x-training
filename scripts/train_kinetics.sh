#!/bin/sh
python train.py --data /data_ssd/processed/ --deterministic --epochs 200 --compress policies/schedule_kinetics.yaml --optimizer Adam --lr 0.001 --wd 0.001 --model ai85actiontcn --use-bias --dataset Kinetics400 --device MAX78000 --batch-size 32 --qat-policy policies/qat_policy.yaml --validation-split 0 "$@"
