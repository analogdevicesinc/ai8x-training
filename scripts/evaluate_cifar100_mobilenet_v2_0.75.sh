#!/bin/sh
python train.py --model ai87netmobilenetv2cifar100_m0_75 --dataset CIFAR100 --evaluate --device MAX78002 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-cifar100-mobilenet-v2-0.75-qat8-q.pth.tar -8 --use-bias "$@"
