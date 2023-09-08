#!/bin/sh
python train.py --deterministic --print-freq 200 --epochs 100 --optimizer Adam --lr 0.001 --model ai85tinierssd --use-bias --momentum 0.9 --weight-decay 5e-4 --dataset SVHN_74 --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params_svhn.yaml --batch-size 16 --qat-policy policies/qat_policy_svhn.yaml --validation-split 0 "$@"
