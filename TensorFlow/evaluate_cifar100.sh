#!/bin/sh
./evaluate.py --onnx-file export/cifar100/saved_model.onnx --dataset cifar100 --inputs-as-nchw $@
./evaluate.py --onnx-file export/cifar100/saved_model_dq.onnx --dataset cifar100 --inputs-as-nchw $@
