#!/bin/bash
./train.py --epochs 100 --deterministic --compress schedule-afsk.yaml --model ai85afsknet --dataset AFSK --confusion --device MAX78000 --embedding "$@"
