#!/bin/sh
python train.py --model ai85faceidnet_112 --dataset VGGFace2_FaceID --kd-student-wt 0 --kd-distill-wt 1  --kd-teacher ir_152 --kd-resume pretrained/ir152_dim64/best.pth.tar --kd-relationbased --evaluate --device MAX78000 --exp-load-weights-from ../ai8x-synthesis/trained/ai85-faceid_112-qat-q.pth.tar -8 --use-bias --save-sample 10 --slice-sample "$@"
