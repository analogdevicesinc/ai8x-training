#!/bin/sh
./cnn-gen.py --timeout 8 --debug --autogen tests --top-level cnn --apb-base 0x50100000 --queue-name lowp -L --test-dir tests -p 2 -p 2 -p 1 -p 1 -x 0 -x 4 -x 0  -x 4 --pool-stride 0 --pool-stride 2 --pool-stride 0 --pool-stride 4 -a 0 -a 0 -a 0 -a 1 -r 1 -r 1 -r 1 -r 1 -o 0x2000 -o 0 -o 0x2000 -o 0 -c 0 -c 4 -c 4 -c 4 --verify-writes --stop-after 1 --prefix fashion
