#!/bin/sh
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule.yaml --model ai84net20 --dataset SpeechCom_20 --confusion