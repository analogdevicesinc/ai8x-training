#!/bin/sh
./evaluate.py --onnx-file export/kws20/saved_model.onnx --dataset kws20 "$@"
./evaluate.py --onnx-file export/kws20/saved_model_dq.onnx --dataset kws20 "$@"
