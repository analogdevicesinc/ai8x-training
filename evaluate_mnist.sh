#!/bin/sh
./train.py --model ai84net5 --dataset MNIST --confusion --evaluate --ai84 --exp-load-weights-from trained/ai84-mnist.pth.tar -8
