#!/bin/sh
./train.py --epochs 200 --deterministic --compress schedule.yaml --model ai85netextrasmall --dataset MNIST --confusion --param-hist --pr-curves --embedding --device MAX78000 "$@"
