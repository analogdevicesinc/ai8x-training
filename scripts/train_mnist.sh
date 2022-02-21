#!/bin/sh
python train.py --lr 0.1 --optimizer SGD --epochs 200 --deterministic --compress policies/schedule.yaml --model ai85net5 --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"
