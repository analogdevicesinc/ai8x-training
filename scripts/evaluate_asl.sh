#!/bin/sh
python3 train.py --model ai85aslnet --dataset asl_big --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-asl-qat8-q.pth.tar -8 --device MAX78000 --use-bias "$@"
