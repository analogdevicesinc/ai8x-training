#!/bin/sh
python train.py --epochs 100 --batch_size 64 --optimizer Adam --lr 0.0003 --model cifar10_model --dataset cifar10 --save-sample 1431 $@
python -m tf2onnx.convert --saved-model export/cifar10 --opset 10 --output export/cifar10/saved_model.onnx
