#!/bin/sh
python train.py --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-faceid.yaml --model ai85faceidnet --dataset FaceID --batch-size 100 --device MAX78000 --regression --print-freq 250 "$@"
