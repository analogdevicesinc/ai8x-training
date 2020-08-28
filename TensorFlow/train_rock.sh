#!/bin/sh
python train.py --epochs 100 --batch_size 32 --optimizer Adam --lr 0.001 --model rock_model --dataset rock --save-sample 101 $@
python -m tf2onnx.convert --saved-model export/rock --inputs-as-nchw input_1:0 --opset 10 --output export/rock/saved_model.onnx

