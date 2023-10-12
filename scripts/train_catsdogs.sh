#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-catsdogs.yaml --qat-policy policies/qat_policy_cd.yaml --model ai85cdnet --dataset cats_vs_dogs --confusion --param-hist --embedding --device MAX78000 "$@"
