#!/bin/sh
python test.py
python -m tf2onnx.convert --saved-model saved_model --inputs-as-nchw input_1:0 --opset 10 --output saved_model/saved_model.onnx
