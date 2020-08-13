#!/bin/sh
python evaluate.py --onnx-file export/cifar10/saved_model.onnx --dataset cifar10
python evaluate.py --onnx-file export/cifar10/saved_model_quant.onnx --dataset cifar10

