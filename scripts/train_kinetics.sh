#!/bin/sh
python train.py --epochs 200 --batch-size 32 --optimizer Adam --lr 0.001 --wd 0.001 --use-bias --deterministic --model ai85actiontcn --dataset Kinetics400 --compress policies/schedule_kinetics.yaml --qat-policy policies/qat_policy.yaml --device MAX78000 --validation-split 0 "$@"
