#!/bin/sh
python train.py --model ai85simplenetwide2x --dataset CIFAR100 --confusion --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar100-simplenetwide2x-qat-mixed-q.pth.tar -8 --use-bias "$@"
