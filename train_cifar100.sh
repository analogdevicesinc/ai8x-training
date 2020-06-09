#!/bin/sh
./train.py --epochs 600 --deterministic --optimizer Adam --lr 0.00016 --compress schedule-cifar100.yaml --model ai85simplenet --dataset CIFAR100 --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 $@
