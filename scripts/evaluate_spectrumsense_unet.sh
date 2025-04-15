#!/bin/sh
python train.py --deterministic --model ai85unetlarge --out-fold-ratio 4 --dataset SpectrumSense_s352_c2 --device MAX78000 --qat-policy policies/qat_policy_spectrumsense.yaml --use-bias --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-spectrumsense-unet-large-q.pth.tar "$@"
