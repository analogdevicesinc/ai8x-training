#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model rsresnet4 --dataset FashionMNIST --confusion
