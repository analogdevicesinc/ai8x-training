#!/bin/sh
helpFunction()
{
   echo ""
   echo "Usage: $0 -b numBits"
   echo "\t-b Number of bits for quantization"
   exit 1 # Exit script after printing help
}

while getopts "b:" opt
do
   case "$opt" in
      b ) numBits="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$numBits" ]
then
   echo "Missing number of bits";
   helpFunction
fi

./train.py --epochs 600 --deterministic --optimizer Adam --lr 0.00016 --compress schedule-cifar100-qa"$numBits"bits.yaml --model ai85simplenet --dataset CIFAR100 --device MAX78000 --batch-size 32 --print-freq 100 --validation-split 0 $@
