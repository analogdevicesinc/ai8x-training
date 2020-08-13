#!/bin/sh
python evaluate.py --onnx-file export/mnist/saved_model.onnx --dataset mnist
python evaluate.py --onnx-file export/mnist/saved_model_quant.onnx --dataset mnist
