#!/bin/sh
./train.py --epochs 100 --optimizer Adam --lr 0.005 --model ai85unet_v3 --dataset CamVidAll --device MAX78000 --batch-size 100 --qat-policy qat_policy_camvid.yaml --use-bias --data /data/raw/CamVid_All/ --validation-split 0 "$@"
