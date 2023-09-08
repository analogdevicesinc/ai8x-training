#!/bin/sh
python train.py --model ai85tinierssdface --dataset VGGFace2_FaceDetection --qat-policy policies/qat_policy_facedet.yaml --use-bias --obj-detection --obj-detection-params parameters/obj_detection_params_facedet.yaml --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-facedet-tinierssd-qat8-q.pth.tar -8 --device MAX78000 "$@"
