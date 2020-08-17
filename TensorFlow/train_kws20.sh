#!/bin/sh
python train.py --epochs 100 --batch_size 128 --optimizer Adam --lr 0.0003 --model kws20_model --dataset kws20 --save-sample 7528 $@
python -m tf2onnx.convert --saved-model export/kws20 --opset 10 --output export/kws20/saved_model.onnx
