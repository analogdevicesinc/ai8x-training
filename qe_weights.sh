#!/bin/sh
./train.py --model sresnet4 --dataset FashionMNIST --confusion --evaluate --resume-from logs/2019.05.09-181300/checkpoint.pth.tar --kernel-stats
