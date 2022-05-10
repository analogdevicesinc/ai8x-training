#!/bin/sh
python3 train.py --deterministic --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-imagenet-effnet2.yaml --model ai87imageneteffnetv2 --dataset ImageNet --device MAX78002 --batch-size 256 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_imagenet.yaml "$@"
