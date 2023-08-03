#!/bin/sh
python3 train.py --regression --deterministic --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --compress policies/schedule-imagenet-effnet2.yaml --model bayer2rgbnet --dataset ImageNet_Bayer --device MAX78002 --batch-size 128 --print-freq 100 --validation-split 0 --use-bias --qat-policy policies/qat_policy_imagenet.yaml --data /data_ssd "$@"
