#!/bin/sh
python evaluate.py --onnx-file export/cifar10/saved_model.onnx --dataset cifar10 --inputs-as-nchw
python evaluate.py --onnx-file export/cifar10/saved_model_dq.onnx --dataset cifar10 --inputs-as-nchw

