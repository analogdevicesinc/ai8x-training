#!/bin/sh
./evaluate.py --onnx-file export/fashionmnist/saved_model.onnx --dataset fashionmnist $@
./evaluate.py --onnx-file export/fashionmnist/saved_model_dq.onnx --dataset fashionmnist $@
