#!/bin/sh
python train.py --optimizer SGD --epochs 200 --deterministic --compress policies/schedule.yaml --model ai85net6 --dataset CIFAR10 --confusion --device MAX78000 --lr 0.01 --param-hist "$@"
