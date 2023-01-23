#!/bin/sh
python train.py --deterministic --model ai85unetlarge --out-fold-ratio 4 --dataset AISegment_352 --device MAX78000 --qat-policy policies/qat_policy_aisegment.yaml --use-bias --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-aisegment-unet-large-q.pth.tar --print-freq 200 "$@"
