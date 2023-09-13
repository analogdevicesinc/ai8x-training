#!/bin/sh
python train.py --regression --deterministic --epochs 100 --optimizer Adam --lr 0.01 --model bayer2rgbnet --dataset ImageNet_Bayer --device MAX78000 --batch-size 64 --print-freq 1000 --validation-split 0 --qat-policy policies/qat_policy_bayer2rgbnet.yaml "$@"
