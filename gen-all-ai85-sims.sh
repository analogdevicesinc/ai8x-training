#!/bin/sh
# ./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file mnist-chw.yaml --fc-layer --embedded-code
# ./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code
# ./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix cifar-bias --checkpoint-file trained/ai84-cifar10-bias.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code

./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-cifar-bias --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file cifar10-hwc.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen tests --top-level cnn -L --test-dir tests --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 --ai85
