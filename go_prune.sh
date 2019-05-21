#!/bin/sh
./train.py --epochs 300 --deterministic --compress prune.yaml --model sresnet4 --dataset FashionMNIST --confusion --resume-from logs/2019.05.09-181300/checkpoint.pth.tar
