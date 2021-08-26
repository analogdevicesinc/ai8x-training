#!/bin/sh
python3 train.py --model ai84net5 --dataset FashionMNIST --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai84-fashionmnist.pth.tar -8 "$@"
