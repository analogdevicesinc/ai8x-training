#!/bin/sh
python train.py --deterministic --print-freq 1 --model ai85autoencoder --dataset MotorDataVoyager4_ForTrain --batch-size 12 --regression --device MAX78000 --qat-policy policies/qat_policy_autoencoder.yaml --use-bias --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-autoencoder-motordatavoyager4-qat-q.pth.tar -8 --print-freq 1 "$@"


