#!/bin/sh
python evaluate.py --onnx-file export/fashionmnist/saved_model.onnx --dataset fashionmnist
python evaluate.py --onnx-file export/fashionmnist/saved_model_dq.onnx --dataset fashionmnist
