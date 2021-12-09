#!/bin/sh
python train.py --model ai85netextrasmall --dataset MNIST --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-mnist-extrasmall-qat8-q.pth.tar -8 --device MAX78000 "$@"
