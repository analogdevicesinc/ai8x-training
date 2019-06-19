#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model ai84netextrasmall --dataset MNIST --confusion
