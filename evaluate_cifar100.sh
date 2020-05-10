#!/bin/sh
./train.py --model ai85simplenet --dataset CIFAR100 --confusion --evaluate --device 85 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar100.pth.tar -8
