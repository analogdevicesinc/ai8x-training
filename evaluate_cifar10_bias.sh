#!/bin/sh
./train.py --model ai84net5 --dataset CIFAR10 --confusion --evaluate --ai84 --exp-load-weights-from trained/ai84-cifar10-bias.pth.tar -8 --use-bias
