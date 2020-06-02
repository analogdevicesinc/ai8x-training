#!/bin/sh
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnet --dataset FaceID --batch-size 100 --device 85 --regression
