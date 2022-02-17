#!/bin/sh
python train.py --model ai85net5 --dataset MNIST --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-mnist-qat8-q.pth.tar -8 --device MAX78000 "$@"
