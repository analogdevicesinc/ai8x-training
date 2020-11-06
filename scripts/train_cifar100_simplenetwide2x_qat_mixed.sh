#!/bin/sh
./train.py --epochs 600 --optimizer Adam --lr 0.00032 --compress schedule-cifar100.yaml --model ai85simplenetwide2x --dataset CIFAR100 --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 --qat-policy qat_policy_cifar100.yaml "$@"
