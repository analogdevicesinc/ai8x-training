CC=gcc
CFLAGS=-I. -ICMSIS_5/CMSIS/Core/Include -ICMSIS_5/CMSIS/NN/Include -ICMSIS_5/CMSIS/DSP/Include -Wall

.PHONY: all
all: main

main: main.o arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_softmax_s8p7_q15.o arm_softmax_s8p7_q15_frac.o arm_fully_connected_q7_s8p7_opt.o arm_convolve_HWC_q7_fast.o
	$(CC) -o main main.o arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_softmax_s8p7_q15.o arm_softmax_s8p7_q15_frac.o arm_fully_connected_q7_s8p7_opt.o arm_convolve_HWC_q7_fast.o

DEPS=sampledata.h weights.h tornadocnn.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean
clean:
	rm main main.o arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_softmax_s8p7_q15.o arm_softmax_s8p7_q15_frac.o arm_fully_connected_q7_s8p7_opt.o arm_convolve_HWC_q7_fast.o
