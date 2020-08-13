#!/bin/sh
python evaluate.py --onnx-file export/cifar100/saved_model.onnx --dataset cifar100
python evaluate.py --onnx-file export/cifar100/saved_model_quant.onnx --dataset cifar100
