#!/bin/sh
echo "-----------------------------"
echo "Training cats&dogs model"
./train_catsdogs.sh
echo "-----------------------------"
echo "Training mnist model"
./train_mnist.sh
echo "-----------------------------"
echo "Training mnist_extrasmall model"
./train_mnist_extrasmall.sh
echo "-----------------------------"
echo "Training cifar10 model"
./train_cifar10.sh
echo "-----------------------------"
echo "Training cifar100 model with no QAT"
./train_cifar100.sh
echo "-----------------------------"
echo "Training cifar100 model with 8-bit QAT"
./train_cifar100_qat8.sh
echo "-----------------------------"
echo "Training cifar100 model with mixed bit width QAT"
./train_cifar100_qat_mixed.sh
echo "-----------------------------"
echo "Training cifar100_residual model"
./train_cifar100_ressimplenet.sh
echo "-----------------------------"
echo "Training kws20 model"
./train_kws20.sh
echo "-----------------------------"
echo "Training kws20_v2 model"
./train_kws20_v2.sh
echo "-----------------------------"
echo "Training faceid model"
./train_faceid.sh

