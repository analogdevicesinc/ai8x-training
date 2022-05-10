#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.0006 --wd 0 --deterministic --compress policies/schedule_kws20_v2.yaml --model ai85kws20netv2 --dataset KWS_20 --confusion --device MAX78000 "$@"
