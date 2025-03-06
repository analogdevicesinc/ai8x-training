#!/bin/sh
python train.py --deterministic --regression --print-freq 1 --epochs 400 --optimizer Adam --lr 0.001 --wd 0 --model ai85autoencoder --use-bias --dataset MotorDataVoyager4_ForTrain --device MAX78000 --batch-size 12 --validation-split 0 --show-train-accuracy full --qat-policy policies/qat_policy_autoencoder.yaml "$@"
