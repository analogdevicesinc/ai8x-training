#!/bin/sh
python3 train.py --epochs 200 --deterministic --compress schedule.yaml --model ai84net5 --dataset CIFAR10 --confusion --use-bias --param-hist "$@"
