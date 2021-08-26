#!/bin/bash
python3 train.py --epochs 100 --deterministic --compress schedule-afsk.yaml --model ai85afsknet --dataset AFSK --confusion --device MAX78000 --embedding "$@"
