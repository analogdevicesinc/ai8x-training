#!/bin/sh
cp Device/*.c ../AI84TK/Firmware/trunk/Applications/EvKitExamples/Common/
cp Device/tornadocnn.h ../AI84TK/Firmware/trunk/Applications/EvKitExamples/Common/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file mnist-chw.yaml --fc-layer --embedded-code
cp demos/MNIST/* ../AI84TK/Firmware/trunk/Applications/EvKitExamples/MNIST/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file cifar10-hwc.yaml --fc-layer --embedded-code
cp demos/CIFAR-10/* ../AI84TK/Firmware/trunk/Applications/EvKitExamples/CIFAR-10/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST-ExtraSmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code
cp demos/MNIST-ExtraSmall/* ../AI84TK/Firmware/trunk/Applications/EvKitExamples/MNIST-ExtraSmall/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST-Small --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file mnist-chw-smallnet.yaml --fc-layer --embedded-code
cp demos/MNIST-Small/* ../AI84TK/Firmware/trunk/Applications/EvKitExamples/MNIST-Small/
