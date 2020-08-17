#!/bin/sh
python train.py --epochs 100 --batch_size 256 --optimizer Adam --lr 0.001 --model fashionmnist_model --dataset fashionmnist --save-sample 1234 --save-sample-per-class $@
python -m tf2onnx.convert --saved-model export/fashionmnist --opset 10 --output export/fashionmnist/saved_model.onnx

