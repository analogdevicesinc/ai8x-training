#!/bin/sh
./train.py --model ai85audionet --dataset KWS_20 --data ./data --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-kws20.pth.tar -8 --device 85