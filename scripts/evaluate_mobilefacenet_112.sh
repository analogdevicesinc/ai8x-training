#!/bin/sh
python train.py --model ai87netmobilefacenet_112 --dataset VGGFace2_FaceID --kd-student-wt 0 --kd-distill-wt 1  --kd-teacher ir_152 --kd-resume pretrained/ir152_dim64/best.pth.tar --kd-relationbased --evaluate --device MAX78002 --exp-load-weights-from ../ai8x-synthesis/trained/ai87-mobilefacenet-112-qat-q.pth.tar -8 --use-bias --save-sample 10 --slice-sample "$@"
