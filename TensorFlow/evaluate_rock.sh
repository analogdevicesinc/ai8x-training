#!/bin/sh
python evaluate.py --onnx-file export/rock/saved_model.onnx --dataset rock --inputs-as-nchw
python evaluate.py --onnx-file export/rock/saved_model_dq.onnx --dataset rock --inputs-as-nchw
