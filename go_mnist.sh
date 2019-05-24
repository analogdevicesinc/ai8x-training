#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model ai84net5 --dataset MNIST --confusion -1
