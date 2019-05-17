#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model sresnet4 --dataset FashionMNIST --confusion
