#!/bin/sh
./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-cifar-bias --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file cifar10-hwc.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file test-ai85-cifar10-hwc-quant4.yaml --ai85

./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file mnist-chw-extrasmallnet.yaml --stop-after 2 --ai85
./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-q4-16x16avgpool --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file test-ai85-cifar10-hwc-16x16avgpool.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-3x3s2p2avgpool --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file test-ai85-cifar10-hwc-3x3s2p2avgpool.yaml --stop-after 0 --ai85

./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir demos --prefix ai85-3x3s1avgpool --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file test-pooling3x3s1.yaml --stop-after 0 --ai85

