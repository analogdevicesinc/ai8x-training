#!/bin/sh
./train.py --model ai84net5 --dataset FashionMNIST --confusion --evaluate --ai84 --exp-load-weights-from ai84.pth.tar -8 -i
