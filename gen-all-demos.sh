#!/bin/sh
./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file mnist-chw.yaml --fc-layer --embedded-code
./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code
