#!/bin/sh
python train.py --model ai85faceidnet --dataset FaceID --regression --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-faceid-qat8-q.pth.tar -8 --device MAX78000 "$@"
