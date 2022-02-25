#!/bin/sh
python train.py --deterministic --model ai85unetlarge --dataset CamVid_s352_c3 --device MAX78000 --qat-policy policies/qat_policy_camvid.yaml --use-bias --evaluate -8 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-camvid-unet-large-q.pth.tar "$@"
