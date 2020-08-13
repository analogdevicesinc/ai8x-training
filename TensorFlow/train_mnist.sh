#!/bin/sh
python train.py --epochs 100 --batch_size 256 --optimizer Adam --lr 0.001 --model mnist_model --dataset mnist --save-sample 1 $@
python -m tf2onnx.convert --saved-model export/mnist --opset 10 --output export/mnist/saved_model.onnx
