#!/bin/sh
./train.py --epochs 100 --batch_size 32 --optimizer Adam --lr 0.001 --model rock_model --dataset rock --save-sample 101 --save-sample-per-class --channel-first "$@"
./convert.py --saved-model export/rock --inputs-as-nchw input_1:0 --opset 10 --output export/rock/saved_model.onnx
