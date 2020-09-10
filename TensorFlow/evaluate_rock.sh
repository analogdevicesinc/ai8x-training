#!/bin/sh
./evaluate.py --onnx-file export/rock/saved_model.onnx --dataset rock --inputs-as-nchw $@
./evaluate.py --onnx-file export/rock/saved_model_dq.onnx --dataset rock --inputs-as-nchw $@
