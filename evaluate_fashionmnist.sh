#!/bin/sh
./train.py --model ai84net5 --dataset FashionMNIST --confusion --evaluate --ai84 --exp-load-weights-from trained/ai84-fashionmnist.pth.tar -8
