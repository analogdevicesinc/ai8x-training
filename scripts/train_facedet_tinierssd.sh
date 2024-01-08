#!/bin/sh
python train.py --deterministic --print-freq 100 --epochs 3 --optimizer Adam --lr 1e-3 --wd 5e-4 --model ai85tinierssdface --use-bias --momentum 0.9 --dataset VGGFace2_FaceDetection --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params_facedet.yaml --batch-size 100 --qat-policy policies/qat_policy_facedet.yaml --validation-split 0.1 "$@"
