#!/bin/sh
./train.py --model ai85net6 --dataset CIFAR10 --confusion --evaluate --device 85 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar10-1x1.pth.tar -8
