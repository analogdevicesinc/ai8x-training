#!/bin/sh
./train.py --epochs 200 --optimizer Adam --lr 0.001 --deterministic --compress schedule_audionet_folded.yaml --model ai85audionet --dataset SpeechComFolded1D_20 --confusion --data /data/ml --device 85
