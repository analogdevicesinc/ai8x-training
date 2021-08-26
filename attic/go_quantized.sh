#!/bin/sh
python3 train.py --epochs 200 --deterministic --compress quant_train_ai84.yaml --model ai84net5 --dataset FashionMNIST --confusion "$@"
