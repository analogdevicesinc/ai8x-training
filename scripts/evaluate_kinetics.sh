#!/bin/sh
python train.py --model ai85actiontcn --dataset Kinetics400 --batch-size 32 --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kinetics-qat8-q.pth.tar -8 --device MAX78000 --use-bias "$@"
