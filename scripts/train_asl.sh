#!/bin/sh
python3 train.py --epochs 100 --optimizer Adam --lr 0.00030 --batch-size 256 --deterministic --compress schedule-asl.yaml --model ai85aslnet --dataset asl_big --confusion --device MAX78000 --use-bias "$@"
