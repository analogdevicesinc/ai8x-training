#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2020-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Script that is used for fusing/folding batchnorm layers onto conv2d layers.
"""

import argparse

import torch


def bn_fuser(state_dict):
    """
    Fuses the BN parameters and returns a new statedict
    """
    dict_keys = state_dict.keys()
    set_convbn_layers = set()
    for dict_key in dict_keys:
        if dict_key.endswith('.bn.running_mean'):
            set_convbn_layers.add(dict_key.rsplit('.', 3)[0])

    for layer in set_convbn_layers:
        if layer + '.op.weight' in state_dict:
            conv_key = layer + '.op'
        else:
            conv_key = layer + '.conv2d'  # Compatibility with older checkpoints
        w_key = conv_key + '.weight'
        b_key = conv_key + '.bias'

        bn_key = layer + '.bn'
        r_mean_key = bn_key + '.running_mean'
        r_var_key = bn_key + '.running_var'
        beta_key = bn_key + '.weight'
        gamma_key = bn_key + '.bias'
        batches_key = bn_key + '.num_batches_tracked'

        w = state_dict[w_key]
        device = state_dict[w_key].device
        if b_key in state_dict:
            b = state_dict[b_key]
        else:
            b = torch.zeros(w.shape[0], device=device)
        if r_mean_key in state_dict:
            r_mean = state_dict[r_mean_key]
        if r_var_key in state_dict:
            r_var = state_dict[r_var_key]
            r_std = torch.sqrt(r_var + 1e-20)
        if beta_key in state_dict:
            beta = state_dict[beta_key]
        else:
            beta = torch.ones(w.shape[0], device=device)
        if gamma_key in state_dict:
            gamma = state_dict[gamma_key]
        else:
            gamma = torch.zeros(w.shape[0], device=device)

        beta = 0.25 * beta
        gamma = 0.25 * gamma

        w_new = w * (beta / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
        b_new = (b - r_mean)/r_std * beta + gamma

        state_dict[w_key] = w_new
        state_dict[b_key] = b_new

        if r_mean_key in state_dict:
            del state_dict[r_mean_key]
        if r_var_key in state_dict:
            del state_dict[r_var_key]
        if beta_key in state_dict:
            del state_dict[beta_key]
        if gamma_key in state_dict:
            del state_dict[gamma_key]
        if batches_key in state_dict:
            del state_dict[batches_key]

    return state_dict


def main(args):
    """
    Main function
    """
    inp_path = args.inp_path
    out_path = args.out_path
    out_arch = args.out_arch

    model_params = torch.load(inp_path)
    new_state_dict = bn_fuser(model_params['state_dict'])
    model_params['state_dict'] = new_state_dict
    model_params['arch'] = out_arch

    torch.save(model_params, out_path)
    print(f'New checkpoint is saved to: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp_path', type=str, required=True, default=12,
                        help='Input checkpoint path')
    parser.add_argument('-o', '--out_path', type=str, required=True, default=20,
                        help='Fused output checkpoint path')
    parser.add_argument('-oa', '--out_arch', type=str, required=True, default=20,
                        help='Output arch name')
    arguments = parser.parse_args()
    main(arguments)
