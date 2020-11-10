#!/bin/sh
./train.py --epochs 200 --deterministic --seed 1 --compress schedule.yaml --model ai85net5 --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 --qat-policy qat_policy_mnist.yaml "$@"
