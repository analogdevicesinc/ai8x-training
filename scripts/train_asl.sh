#!/bin/sh
#./train.py --epochs 100 --optimizer Adam --lr 0.00014 --batch-size 256 --deterministic --compress schedule-rps.yaml --model ai85nascifarnet --dataset asl_big --confusion --device MAX78000 --use-bias "$@"
#export PATH="/home/narendratanganiya/.pyenv/bin:$PATH"
#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"

#./train.py --epochs 100 --optimizer Adam --lr 0.00032 --compress schedule-rps.yaml --model ai85nascifarnet --dataset asl_all --confusion --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 --qat-policy None "$@"

./train.py --epochs 100 --optimizer Adam --lr 0.00030 --batch-size 256 --deterministic --compress schedule-rps.yaml --model ai85rpsnet --dataset asl_big --confusion --device MAX78000 --use-bias "$@"