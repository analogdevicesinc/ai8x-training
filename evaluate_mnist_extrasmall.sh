#!/bin/sh
./train.py --model ai85netextrasmall --dataset MNIST --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-mnist-extrasmall.pth.tar -8 --device MAX78000 "$@"
