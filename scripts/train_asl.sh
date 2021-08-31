#!/bin/sh
git update-index --add --chmod=+x train_asl.sh
python3 train.py --epochs 100 --optimizer Adam --lr 0.00030 --batch-size 256 --deterministic --compress schedule-rps.yaml --model ai85rpsnet --dataset asl_big --confusion --device MAX78000 --use-bias "$@"
