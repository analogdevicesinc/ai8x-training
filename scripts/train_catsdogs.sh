#!/bin/sh
python train.py --epochs 250 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-catsdogs.yaml --model ai85cdnet --dataset cats_vs_dogs --confusion --param-hist --embedding --device MAX78000 "$@"
