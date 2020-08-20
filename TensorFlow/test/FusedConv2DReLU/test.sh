#!/bin/sh
python test.py
python -m tf2onnx.convert --saved-model saved_model --opset 10 --output saved_model/saved_model.onnx
