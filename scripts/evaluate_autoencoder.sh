#!/bin/sh
python train.py --deterministic --model ai85autoencoder --dataset SampleMotorDataLimerick_ForEvalWithSignal --regression --device MAX78000 --qat-policy policies/qat_policy_autoencoder.yaml --use-bias --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-autoencoder-samplemotordatalimerick-qat-q.pth.tar -8 --print-freq 1 "$@"
