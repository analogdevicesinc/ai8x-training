#!/bin/sh
./train.py --epochs 200 --deterministic --compress quant_train.yaml --model sresnet4 --dataset FashionMNIST --confusion
