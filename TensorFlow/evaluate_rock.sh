#!/bin/sh
python evaluate.py --onnx-file export/rock/saved_model.onnx --dataset rock
python evaluate.py --onnx-file export/rock/saved_model_quant.onnx --dataset rock
