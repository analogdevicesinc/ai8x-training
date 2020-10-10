#!/bin/sh
./train.py --model ai85net5 --dataset MNIST --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-mnist.pth.tar -8 --device MAX78000 "$@"
