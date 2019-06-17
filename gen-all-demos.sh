#!/bin/sh
./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file mnist-chw.yaml --fc-layer --embedded-code
./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code
cp Device/*.c ../AI84TK/Firmware/trunk/Applications/EvKitExamples/Common/
cp Device/tornadocnn.h ../AI84TK/Firmware/trunk/Applications/EvKitExamples/Common/
cp demos/MNIST/* ../AI84TK/Firmware/trunk/Applications/EvKitExamples/MNIST/
cp demos/CIFAR-10/* ../AI84TK/Firmware/trunk/Applications/EvKitExamples/CIFAR-10/
