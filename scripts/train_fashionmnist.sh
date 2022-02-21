#!/bin/sh
python train.py --lr 0.1 --optimizer SGD --epochs 200 --deterministic --compress policies/schedule.yaml --model ai84net5 --dataset FashionMNIST --confusion "$@"
