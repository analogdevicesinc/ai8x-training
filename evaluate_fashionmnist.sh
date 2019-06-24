#!/bin/sh
./train.py --model ai84net5 --dataset FashionMNIST --confusion --evaluate --ai84 --exp-load-weights-from ../ai8x-synthesis/trained/ai84-fashionmnist.pth.tar -8
