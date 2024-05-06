#!/bin/sh
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=auto --no-python "$@"
