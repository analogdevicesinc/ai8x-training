#!/bin/sh

./train.py --model ai85rpsnet --dataset asl_big --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-asl01-chw.pth.tar -8 --device MAX78000 --use-bias "$@"

#./train.py --model ai85nascifarnet --dataset asl_all --confusion --evaluate --exp-load-weights-from ./logs/2021.07.06-132039/checkpoint.pth.tar --device MAX78000 "$@"
