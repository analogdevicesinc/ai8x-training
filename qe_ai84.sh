#!/bin/sh
./train.py --model sresnet4 --dataset FashionMNIST --confusion --evaluate --qe --qe-config-file post_train_ai84.yaml --ai84 --resume-from logs/2019.05.09-181300/checkpoint.pth.tar
