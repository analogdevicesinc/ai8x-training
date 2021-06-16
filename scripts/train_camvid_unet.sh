#!/bin/sh
./train.py --deterministic --epochs 100 --optimizer Adam --lr 0.001 --model ai85unetmedium --dataset CamVid_3 --device MAX78000 --batch-size 100 --qat-policy qat_policy_camvid.yaml --use-bias "$@"
