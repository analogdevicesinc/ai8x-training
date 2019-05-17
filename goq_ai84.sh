#!/bin/sh
./train.py --epochs 200 --deterministic --compress quant_train_ai84.yaml --model sresnet4 --dataset FashionMNIST --confusion
