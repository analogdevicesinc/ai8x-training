#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Load contents of quantized checkpoint files and save them in a format usable for AI84
"""
import torch
import argparse
from distiller.apputils.checkpoint import get_contents_table
from range_linear_ai84 import pow2_round

CONV_SCALE_BITS = 3
CONV_WEIGHT_BITS = 5
CONV_CLAMP_BITS = 8
FC_SCALE_BITS = 8
FC_WEIGHT_BITS = 8
FC_CLAMP_BITS = 16


def convert_checkpoint(chkpt_file, output_file, args):
    print("Converting checkpoint file", chkpt_file, "to", output_file)
    checkpoint = torch.load(chkpt_file, map_location='cpu')

    if args.verbose:
        print(get_contents_table(checkpoint))

    if 'quantizer_metadata' not in checkpoint:
        raise RuntimeError("\nNo quantizer_metadata in checkpoint file.")
    del checkpoint['quantizer_metadata']

    if 'state_dict' not in checkpoint:
        raise RuntimeError("\nNo state_dict in checkpoint file.")

    checkpoint_state = checkpoint['state_dict']

    if args.verbose:
        print("\nModel keys (state_dict):\n{}".format(", ".join(list(checkpoint_state.keys()))))

    new_checkpoint_state = checkpoint_state.copy()

    for i, k in enumerate(checkpoint_state.keys()):
        operation, parameter = k.rsplit(sep='.', maxsplit=1)
        if parameter in ['w_zero_point', 'b_zero_point']:
            if checkpoint_state[k].nonzero().numel() != 0:
                raise RuntimeError(f"\nParameter {k} is not zero.")
            del new_checkpoint_state[k]
        elif parameter in ['weight', 'bias']:
            module, st = operation.rsplit('.', maxsplit=1)
            if st in ['wrapped_module']:
                scale = module + '.' + parameter[0] + '_scale'
                weights = checkpoint_state[k]
                scale = module + '.' + parameter[0] + '_scale'
                fp_scale = checkpoint_state[scale]
                (scale_bits, clamp_bits) = (CONV_SCALE_BITS, CONV_CLAMP_BITS) \
                    if module != 'fc' else (FC_SCALE_BITS, FC_CLAMP_BITS)
                if module not in ['fc']:
                    weights *= pow2_round(fp_scale, scale_bits)
                else:
                    weights = torch.round(weights * fp_scale)
                weights.round().clamp(min=-(2**(clamp_bits-1)), max=2**(clamp_bits-1)-1)

                new_checkpoint_state[module + '.' + parameter] = weights
                del new_checkpoint_state[k]
                del new_checkpoint_state[scale]
        elif parameter in ['base_b_q']:
            del new_checkpoint_state[k]

    if not args.embedded:
        checkpoint['state_dict'] = new_checkpoint_state
        torch.save(checkpoint, output_file)
    else:
        # Create parameters for AI84
        for i, k in enumerate(new_checkpoint_state.keys()):
            print(f'#define {k.replace(".", "_").upper()} \\')
            print(new_checkpoint_state[k].numpy().astype(int))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantized checkpoint to AI84 conversion')
    parser.add_argument('chkpt_file', help='path to the checkpoint file')
    parser.add_argument('output_file', help='path to the output file')
    parser.add_argument('-e', '--embedded', action='store_true', default=False,
                        help='save parameters for embedded (default: rewrite checkpoint)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='verbose mode')
    args = parser.parse_args()
    convert_checkpoint(args.chkpt_file, args.output_file, args)
