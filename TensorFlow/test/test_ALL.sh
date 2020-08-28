#!/bin/sh

cd FusedDense
rm -R saved_model
./test.sh
cd ../FusedConv1D
rm -R saved_model
./test.sh
cd ../FusedConv1D_Conv1D
rm -R saved_model
./test.sh
cd ../FusedConv1D_Conv1D_Dense
rm -R saved_model
./test.sh
cd ../FusedConv1DReLU_bias
rm -R saved_model
./test.sh
cd ../FusedAvgPoolConv1DReLU
rm -R saved_model
./test.sh
cd ../FusedMaxPoolConv1DReLU
rm -R saved_model
./test.sh
cd ../FusedConv1D_Conv2D
rm -R saved_model
./test.sh
cd ../FusedConv2D
rm -R saved_model
./test.sh
cd ../FusedConv2DReLU
rm -R saved_model
./test.sh
cd ../FusedConv2DReLU_bias
rm -R saved_model
./test.sh
cd ../FusedAvgPoolConv2DReLU_bias
rm -R saved_model
./test.sh
cd ../FusedMaxPoolConv2DReLU_bias
rm -R saved_model
./test.sh
cd ../Conv2D_Conv2D_Dense
rm -R saved_model
./test.sh
cd ../SimpleNet
rm -R saved_model
./test.sh
cd ../SimpleNet2
rm -R saved_model
./test.sh
cd ../SimpleNet3
rm -R saved_model
./test.sh
cd ../FusedConv2DTranspose
rm -R saved_model
./test.sh
cd ../TR_Conv2d_Conv2D_TR_Dense
rm -R saved_model
./test.sh
cd ./TR_Conv2d_TR_Dense
rm -R saved_model
./test.sh
cd ..

python testCommandline.py --folder ./ --keyword 'Output(8' --skip$@


