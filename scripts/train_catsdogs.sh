#!/bin/sh
python3 train.py --epochs 200 --optimizer Adam --lr 0.001 --deterministic --compress schedule-catsdogs.yaml --model ai85cdnet --dataset cats_vs_dogs --confusion --param-hist --embedding --device MAX78000 "$@"
