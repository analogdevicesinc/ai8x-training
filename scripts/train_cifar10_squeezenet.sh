#!/bin/sh
python train.py --optimizer SGD --epochs 250 --lr 0.02 --compress policies/schedule_squeezenet_cifar10.yaml --model ai85squeezenet --dataset CIFAR10 --confusion --param-hist --device MAX78000 "$@"
