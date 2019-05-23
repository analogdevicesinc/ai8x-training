#!/bin/sh
./train.py --model ai84net5 --dataset FashionMNIST --confusion --evaluate --resume-from logs/FashionMNIST/checkpoint.pth.tar --kernel-stats
