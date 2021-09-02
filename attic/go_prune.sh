#!/bin/sh
python3 train.py --epochs 300 --deterministic --compress prune.yaml --model ai84net5 --dataset FashionMNIST --confusion --resume-from logs/FashionMNIST/checkpoint.pth.tar "$@"
