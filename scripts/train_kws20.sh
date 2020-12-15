#!/bin/sh
./train.py --epochs 200 --optimizer Adam --lr 0.001 --deterministic --compress schedule_kws20.yaml --model ai85kws20net --dataset KWS_20 --confusion --device MAX78000 "$@"
