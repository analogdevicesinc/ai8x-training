#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model ai85net5 --dataset CIFAR10 --confusion --param-hist --pr-curves --embedding --device MAX78000 $@
