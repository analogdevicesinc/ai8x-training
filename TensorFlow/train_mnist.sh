#!/bin/sh
python train.py --epochs 100 --batch_size 256 --optimizer Adam --lr 0.001 --model mnist_model --dataset mnist --save-sample 1 --save-sample-per-class $@
python -m tf2onnx.convert --saved-model export/mnist --inputs-as-nchw input_1:0 --opset 10 --output export/mnist/saved_model.onnx
