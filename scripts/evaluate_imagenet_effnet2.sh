#!/bin/sh
python3 train.py --model ai87imageneteffnetv2 --dataset ImageNet --evaluate --device MAX78002 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-imagenet-effnet2-q.pth.tar --save-sample 100 -8 --use-bias "$@"
