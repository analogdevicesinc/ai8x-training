#!/bin/sh
./cnn-gen.py -v --top-level fashion0 --test-dir tests -p 2 -p 1 -p 1 -p 1 -x 0 -x 4 -x 0  -x 4 --pool-stride 0 --pool-stride 2 --pool-stride 0 --pool-stride 4 -a 0 -a 0 -a 0 -a 1 -r 1 -r 1 -r 1 -r 1
