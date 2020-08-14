#!/bin/sh
python train.py --epochs 100 --batch_size 32 --optimizer Adam --lr 0.0003 --model cifar100_model --dataset cifar100 --save-sample 1193 $@
python -m tf2onnx.convert --saved-model export/cifar100 --opset 10 --output export/cifar100/saved_model.onnx
