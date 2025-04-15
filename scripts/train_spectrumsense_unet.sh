#!/bin/sh
python train.py --deterministic --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --model ai85unetlarge --out-fold-ratio 4 --use-bias --dataset SpectrumSense_s352_c2 --device MAX78000 --batch-size 50 --qat-policy policies/qat_policy_spectrumsense.yaml --compress policies/schedule-spectrumsense.yaml --validation-split 0  "$@"

