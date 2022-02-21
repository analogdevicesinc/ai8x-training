#!/bin/sh
python train.py --deterministic --epochs 300 --optimizer SGD --lr 0.1 --compress policies/schedule-cifar100-mobilenetv2.yaml --model ai87netmobilenetv2cifar100_m0_75 --dataset CIFAR100 --device MAX78002 --batch-size 128 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_cifar100_mobilenetv2.yaml "$@"
