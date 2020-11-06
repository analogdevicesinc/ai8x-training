#!/bin/sh
./train.py --model ai85net5 --dataset CIFAR10 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar10-qat8-q.pth.tar -8 --device MAX78000 "$@"
