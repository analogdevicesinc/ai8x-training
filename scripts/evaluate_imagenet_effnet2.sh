#!/bin/sh
python train.py --model ai87imageneteffnetv2 --dataset ImageNet --evaluate --device MAX78002 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-imagenet-effnet2-q.pth.tar -8 --use-bias --qat-policy policies/qat_policy_imagenet.yaml "$@"
