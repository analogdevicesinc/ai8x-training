#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --print-freq 100 --deterministic --compress policies/schedule_kws20.yaml --model ai85kws20netv3 --dataset KWS_20_msnoise_mixed --confusion --device MAX78000 "$@"
