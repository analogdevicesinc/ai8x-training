#!/bin/sh
python train.py --deterministic --pr-curves --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --model ai85unetlarge --out-fold-ratio 4 --use-bias --dataset AISegment_352 --device MAX78000 --batch-size 32 --qat-policy policies/qat_policy_aisegment.yaml --compress policies/schedule-aisegment.yaml --validation-split 0 --print-freq 250 "$@"
