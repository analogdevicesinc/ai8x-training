#!/bin/sh
python train.py --model ai85kws20netnas --use-bias --dataset KWS_12_benchmark --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kws12_nas_benchmark-qat8-q.pth.tar -8 --device MAX78000 "$@"
