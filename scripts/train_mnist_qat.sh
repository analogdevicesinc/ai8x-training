#!/bin/sh
python train.py --lr 0.01 --optimizer SGD --epochs 200 --deterministic --seed 1 --compress policies/schedule.yaml --model ai85net5 --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 --qat-policy policies/qat_policy_mnist.yaml "$@"
