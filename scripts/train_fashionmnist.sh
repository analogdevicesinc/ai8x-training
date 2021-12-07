#!/bin/sh
python train.py --epochs 200 --deterministic --compress schedule.yaml --model ai84net5 --dataset FashionMNIST --confusion "$@"
