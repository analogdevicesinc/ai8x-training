#!/bin/sh
./train.py --model ai84net5 --dataset CIFAR10 --confusion --evaluate --qe --qe-config-file post_train_ai84.yaml --ai84 --resume-from logs/CIFAR10/checkpoint.pth.tar -8 -i
