#!/bin/sh
./train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress schedule.yaml --model ai84net7 --dataset SpeechCom --confusion
