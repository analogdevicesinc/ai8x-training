#!/bin/sh
python train.py --model ai87effnetv2 --dataset CIFAR100 --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-cifar100-effnet2-qat8-q.pth.tar -8 --use-bias "$@"
