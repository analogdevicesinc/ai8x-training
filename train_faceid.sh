#!/bin/sh
./train.py --epochs 50 --optimizer Adam --lr 0.001 --deterministic --compress schedule-faceid.yaml --model ai85faceidnet --dataset FaceID --batch-size 100 --use-bias --device 85 $@
