#!/bin/sh
python3 train.py --model ai84net5 --dataset MNIST --confusion --evaluate --qe --qe-config-file post_train_ai84.yaml --resume-from logs/MNIST/checkpoint.pth.tar -1 "$@"
