#!/bin/sh
./train_mnist.sh $@
./train_cifar10.sh $@
./train_cifar100.sh $@
./train_fashionmnist.sh $@
./train_kws20.sh $@
./train_rock.sh $@
