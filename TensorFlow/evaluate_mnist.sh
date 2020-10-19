#!/bin/sh
./evaluate.py --onnx-file export/mnist/saved_model.onnx --dataset mnist "$@"
./evaluate.py --onnx-file export/mnist/saved_model_dq.onnx --dataset mnist "$@"
