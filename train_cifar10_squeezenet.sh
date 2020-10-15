#!/bin/sh
./train.py --epochs 250 --lr 0.02 --compress schedule_squeezenet_cifar10.yaml --model ai85squeezenet --dataset CIFAR10 --confusion --param-hist --device MAX78000 "$@"
