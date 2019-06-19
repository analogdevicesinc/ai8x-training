#!/bin/sh
./cnn-gen.py --top-level cnn --test-dir demos --prefix CIFAR-10-Arm --checkpoint-file trained/ai84-cifar10.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix MNIST-ExtraSmall-Arm --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix MNIST-Small-Arm --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file mnist-chw-smallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
