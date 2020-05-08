#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model ai84net5 --dataset CIFAR100 --confusion --param-hist --pr-curves --embedding