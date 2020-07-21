#!/bin/sh
./train.py --model ai85simplenet --dataset CIFAR100 --confusion --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-cifar100-qat8.pth.tar -8 --qat --qat_num_bits 8 $@
