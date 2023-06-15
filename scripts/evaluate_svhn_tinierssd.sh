#!/bin/sh
python train.py --deterministic --print-freq 200 --batch-size 16 --model ai85tinierssd --use-bias --dataset SVHN_74 --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params_svhn.yaml --qat-policy policies/qat_policy_svhn.yaml --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-svhn-tinierssd-qat8-q.pth.tar --validation-split 0 "$@"
