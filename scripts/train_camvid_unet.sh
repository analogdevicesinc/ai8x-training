#!/bin/sh
python train.py --deterministic --epochs 100 --optimizer Adam --lr 0.001 --model ai85unetlarge --use-bias --dataset CamVid_s352_c3 --device MAX78000 --batch-size 32 --qat-policy qat_policy_camvid.yaml --compress schedule-camvid.yaml --validation-split 0 "$@"
