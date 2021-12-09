#!/bin/sh
python train.py --data /data2/ml/ --deterministic --model ai85unetlarge --dataset AISegment_352 --device MAX78000 --qat-policy qat_policy_aisegment.yaml --use-bias --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-aisegment-unet-large-q.pth.tar --print-freq 200 "$@"
