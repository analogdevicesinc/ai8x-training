#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Load contents of a checkpoint files and save them in a format usable for AI84
"""
import argparse
from functools import partial
import torch
from distiller.apputils.checkpoint import get_contents_table  # pylint: disable=no-name-in-module
import ai84
# import yamlcfg
from range_linear_ai84 import pow2_round

CONV_SCALE_BITS = 8
FC_SCALE_BITS = 8
FC_CLAMP_BITS = 8


def convert_checkpoint(input_file, output_file, arguments):
    """
    Convert checkpoint file or dump parameters for C code
    """
    # Load configuration file
    # cfg, params = yamlcfg.parse(args.config_file)

    print("Converting checkpoint file", input_file, "to", output_file)
    checkpoint = torch.load(input_file, map_location='cpu')

    if arguments.verbose:
        print(get_contents_table(checkpoint))

    if arguments.quantized:
        if 'quantizer_metadata' not in checkpoint:
            raise RuntimeError("\nNo quantizer_metadata in checkpoint file.")
        del checkpoint['quantizer_metadata']

    if 'state_dict' not in checkpoint:
        raise RuntimeError("\nNo state_dict in checkpoint file.")

    checkpoint_state = checkpoint['state_dict']

    if arguments.verbose:
        print("\nModel keys (state_dict):\n{}".format(", ".join(list(checkpoint_state.keys()))))

    new_checkpoint_state = checkpoint_state.copy()

    def avg_max(t):
        dim = 0
        view_dims = [t.shape[i] for i in range(dim + 1)] + [-1]
        tv = t.view(*view_dims)
        avg_min, avg_max = tv.min(dim=-1)[0], tv.max(dim=-1)[0]
        return torch.max(avg_min.mean().abs_(), avg_max.mean().abs_())

    def max_max(t):
        return torch.max(t.min().abs_(), t.max().abs_())

    def mean_n_stds_max_abs(t, n_stds=1):
        if n_stds <= 0:
            raise ValueError(f'n_stds must be > 0, got {n_stds}')
        mean, std = t.mean(), t.std()
        min_val = torch.max(t.min(), mean - n_stds * std)
        max_val = torch.min(t.max(), mean + n_stds * std)
        return torch.max(min_val.abs_(), max_val.abs_())

    def get_const(_):
        return arguments.scale

    # Scale to our fixed point representation using any of four methods
    # The 'magic constant' seems to work best!?? FIXME
    if arguments.clip_mode == 'STD':
        sat_fn = partial(mean_n_stds_max_abs, n_stds=arguments.stddev)
    elif arguments.clip_mode == 'MAX':
        sat_fn = max_max
    elif arguments.clip_mode == 'AVGMAX':
        sat_fn = avg_max
    else:
        sat_fn = get_const
    fc_sat_fn = get_const

    first = True
    for _, k in enumerate(checkpoint_state.keys()):
        operation, parameter = k.rsplit(sep='.', maxsplit=1)
        if parameter in ['w_zero_point', 'b_zero_point']:
            if checkpoint_state[k].nonzero().numel() != 0:
                raise RuntimeError(f"\nParameter {k} is not zero.")
            del new_checkpoint_state[k]
        elif parameter in ['weight', 'bias']:
            if not arguments.quantized:
                module, _ = k.split(sep='.', maxsplit=1)

                if module != 'fc':
                    clamp_bits = ai84.WEIGHT_BITS  # FIXME: Use layer config
                    factor = 2**(clamp_bits-1) * sat_fn(checkpoint_state[k])
                    lower_bound = 0
                    if first:
                        factor /= 2.  # The input layer is [-0.5, +0.5] -- compensate
                        first = False
                else:
                    clamp_bits = arguments.fc
                    lower_bound = 1  # Accomodate ARM q15_t data type when clamping
                    factor = 2**(clamp_bits-1) * fc_sat_fn(checkpoint_state[k])

                if args.verbose:
                    print(k, 'avg_max', avg_max(checkpoint_state[k]),
                          'max', max_max(checkpoint_state[k]),
                          'mean', checkpoint_state[k].mean(),
                          'factor', factor)
                weights = factor * checkpoint_state[k]

                # The scale is different for AI84, and this has to happen before clamping.
                if not arguments.ai85 and module != 'fc' and parameter == 'bias':
                    weights *= 2**(clamp_bits-1)

                # Ensure it fits and is an integer
                weights = weights.clamp(min=-(2**(clamp_bits-1)-lower_bound),
                                        max=2**(clamp_bits-1)-1).round()

                # Save conv biases so PyTorch can still use them to run a model. This needs to be
                # reversed before loading the weights into the AI84/AI85.
                # When multiplying data with weights, 1.0 * 1.0 corresponds to 128 * 128 and we
                # divide the output by 128 to compensate. The bias therefore needs to be multiplied
                # by 128. This depends on the data width, not the weight width, and is therefore
                # always 128.
                if arguments.ai85 and module != 'fc' and parameter == 'bias':
                    weights *= 2**(ai84.DATA_BITS-1)

                # Store modified weight/bias back into model
                new_checkpoint_state[k] = weights
            else:
                # Work on a pre-quantized network
                module, st = operation.rsplit('.', maxsplit=1)
                if st in ['wrapped_module']:
                    scale = module + '.' + parameter[0] + '_scale'
                    weights = checkpoint_state[k]
                    scale = module + '.' + parameter[0] + '_scale'
                    (scale_bits, clamp_bits) = (CONV_SCALE_BITS, ai84.WEIGHT_BITS) \
                        if module != 'fc' else (FC_SCALE_BITS, FC_CLAMP_BITS)
                    fp_scale = checkpoint_state[scale]
                    if module not in ['fc']:
                        # print("Factor in:", fp_scale, "bits", scale_bits, "out:",
                        #       pow2_round(fp_scale, scale_bits))
                        weights *= pow2_round(fp_scale, scale_bits)
                        # Accomodate Arm q15_t/q7_t datatypes
                        weights = weights.clamp(min=-(2**(clamp_bits-1)),
                                                max=2**(clamp_bits-1)-1).round()
                    else:
                        weights = torch.round(weights * fp_scale)
                        weights = weights.clamp(min=-(2**(clamp_bits-1)-1),
                                                max=2**(clamp_bits-1)-1).round()

                    new_checkpoint_state[module + '.' + parameter] = weights
                    del new_checkpoint_state[k]
                    del new_checkpoint_state[scale]
        elif parameter in ['base_b_q']:
            del new_checkpoint_state[k]

    if not arguments.embedded:
        checkpoint['state_dict'] = new_checkpoint_state
        torch.save(checkpoint, output_file)
    else:
        # Create parameters for AI84
        for _, k in enumerate(new_checkpoint_state.keys()):
            print(f'#define {k.replace(".", "_").upper()} \\')
            print(new_checkpoint_state[k].numpy().astype(int))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checkpoint to AI84 conversion')
    parser.add_argument('input', help='path to the checkpoint file')
    parser.add_argument('output', help='path to the output file')
    parser.add_argument('-c', '--config-file', metavar='S',  # required=True,
                        help="YAML configuration file containing layer configuration")
    parser.add_argument('--ai85', action='store_true', default=False,
                        help='enable AI85 features')
    parser.add_argument('-f', '--fc', type=int, default=FC_CLAMP_BITS, metavar='N',
                        help=f'set number of bits for the fully connnected layers '
                             f'(default: {FC_CLAMP_BITS})')
    parser.add_argument('-e', '--embedded', action='store_true', default=False,
                        help='save parameters for embedded (default: rewrite checkpoint)')
    parser.add_argument('-q', '--quantized', action='store_true', default=False,
                        help='work on quantized checkpoint')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='verbose mode')
    parser.add_argument('--clip-mode', default='SCALE',
                        choices=['AVGMAX', 'MAX', 'STD', 'SCALE'],
                        help='saturation clipping for conv2d (default: magic scale)')
    parser.add_argument('--scale', type=float, default=0.85,
                        help='set the scale value for the SCALE mode (default: magic 0.85)')
    parser.add_argument('--stddev', type=float, default=2.0,
                        help='set the number of standard deviations for the STD mode (default: 2)')
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output, args)
