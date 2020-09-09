#!/bin/sh
./train.py --epochs 100 --batch_size 128 --optimizer Adam --lr 0.0003 --model kws20_model --dataset kws20 --save-sample 7528 --save-sample-per-class --channel-first --swap $@
./convert.py --saved-model export/kws20 --opset 10 --output export/kws20/saved_model.onnx
