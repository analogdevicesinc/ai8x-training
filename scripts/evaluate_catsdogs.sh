#!/bin/sh
python train.py --model ai85cdnet --dataset cats_vs_dogs --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-catsdogs-qat8-q.pth.tar -8 --device MAX78000 "$@"
