#!/bin/sh
./train.py --model ai84net5 --dataset FashionMNIST --confusion --evaluate --qe --qe-config-file post_train_ai84.yaml --ai84 --resume-from logs/FashionMNIST/checkpoint.pth.tar -1
