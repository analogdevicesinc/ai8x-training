#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model ai84netsmall --dataset MNIST --confusion
