###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Checkpoint File Routines
"""
import sys
import numpy as np
import torch
import tornadocnn


def load(checkpoint_file, arch, fc_layer, quantization):
    """
    Load weights and biases from `checkpoint_file`. If `arch` is not None and does not match
    the architecuture in the checkpoint file, abort with an error message. If `fc_layer` is
    `True`, configure a single fully connected classification layer.
    `quantization` is a list of expected bit widths for the layer weights (always 8 for AI84).
    This value is checked against the weight inputs.
    In addition to returning weights anf biases, this function configures the network output
    channels and the number of layers.
    """
    weights = []
    bias = []
    fc_weights = []
    fc_bias = []

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    print(f'Reading {checkpoint_file} to configure network weights...')

    if 'state_dict' not in checkpoint or 'arch' not in checkpoint:
        raise RuntimeError("\nNo `state_dict` or `arch` in checkpoint file.")

    if arch and checkpoint['arch'].lower() != arch.lower():
        print(f"Network architecture of configuration file ({arch}) does not match "
              f"network architecture of checkpoint file ({checkpoint['arch']}).")
        sys.exit(1)

    checkpoint_state = checkpoint['state_dict']
    layers = 0
    have_fc_layer = False
    output_channels = []
    for _, k in enumerate(checkpoint_state.keys()):
        operation, parameter = k.rsplit(sep='.', maxsplit=1)
        if parameter in ['weight']:
            module, _ = k.split(sep='.', maxsplit=1)
            if module != 'fc':
                w = checkpoint_state[k].numpy().astype(np.int64)
                assert w.min() >= -(2**(quantization[layers]-1))
                assert w.max() < 2**(quantization[layers]-1)
                if layers == 0:
                    output_channels.append(w.shape[1])  # Input channels
                output_channels.append(w.shape[0])
                weights.append(w.reshape(-1, w.shape[-2], w.shape[-1]))
                # Is there a bias for this layer?
                bias_name = operation + '.bias'
                if bias_name in checkpoint_state:
                    w = checkpoint_state[bias_name].numpy().astype(np.int64) // tornadocnn.BIAS_DIV
                    assert w.min() >= -(2**(quantization[layers]-1))
                    assert w.max() < 2**(quantization[layers]-1)
                    bias.append(w)
                else:
                    bias.append(None)
                layers += 1
            elif have_fc_layer:
                print('The network cannot have more than one fully connected software layer, '
                      'and it must be the output layer.')
                sys.exit(1)
            elif fc_layer:
                w = checkpoint_state[k].numpy().astype(np.int64)
                assert w.min() >= -128 and w.max() <= 127
                fc_weights.append(w)
                # Is there a bias for this layer?
                bias_name = operation + '.bias'
                if bias_name in checkpoint_state:
                    # Do not divide bias for FC
                    w = checkpoint_state[bias_name].numpy().astype(np.int64)
                    assert w.min() >= -128 and w.max() <= 127
                    fc_bias.append(w)
                else:
                    fc_bias.append(None)
                have_fc_layer = True

    return layers, weights, bias, fc_weights, fc_bias, output_channels
