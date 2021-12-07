#!/bin/sh
python train.py --model ai85net6 --dataset CIFAR10 --confusion --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar10-1x1.pth.tar -8 "$@"
