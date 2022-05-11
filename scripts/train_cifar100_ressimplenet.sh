#!/bin/sh
python train.py --epochs 500 --deterministic --optimizer Adam --lr 0.00064 --wd 0 --compress policies/schedule-cifar100-ressimplenet.yaml --model ai85ressimplenet --dataset CIFAR100 --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 "$@"
