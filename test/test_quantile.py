#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Test routine for quantile()
"""
import os
import sys

import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ai8x  # noqa: E402 pylint: disable=wrong-import-position


def calc_quantile(q, x):
    """
    Test ai8x.quantile against torch.quantile() with `num` 1D values.
    """
    def test_method(quantile, values, method):
        q_pytorch = torch.quantile(values, quantile, interpolation=method)
        q_ai8x = ai8x.quantile(values, quantile, method=method)

        print(f'{method+":":10}{q_ai8x}')
        assert torch.isclose(q_ai8x, q_pytorch)

    n = len(x.flatten())
    print(f'\n====== q={q} len={n} ', end='')
    if n < 10:
        print(f'{x.numpy()} ', end='')
    print('======\n'
          'PyTorch:\n'
          f'linear:   {torch.quantile(x, q, interpolation="linear")}\n'
          f'lower:    {torch.quantile(x, q, interpolation="lower")}\n'
          f'higher:   {torch.quantile(x, q, interpolation="higher")}\n'
          f'nearest:  {torch.quantile(x, q, interpolation="nearest")}\n'
          f'midpoint: {torch.quantile(x, q, interpolation="midpoint")}\n')

    print('ai8x:')
    test_method(q, x, 'linear')
    test_method(q, x, 'lower')


def test_quantile():
    """Main program to test ai8x.quantile."""
    calc_quantile(.5, torch.tensor([[1., 2., 3.]]))
    calc_quantile(.5, torch.tensor([[1., 2., 3., 4.]]))
    calc_quantile(.0, torch.tensor([[1., 2., 3.]]))
    calc_quantile(.0, torch.tensor([[1., 2., 3., 4.]]))
    calc_quantile(1., torch.tensor([[1., 2., 3.]]))
    calc_quantile(1., torch.tensor([[1., 2., 3., 4.]]))
    calc_quantile(.75, torch.tensor([[1., 2., 3., 4.]]))
    calc_quantile(.75, torch.tensor([[1., 2., 3.]]))
    calc_quantile(.5, torch.rand((1, 1)))
    calc_quantile(.75, torch.rand((1, 1)))
    calc_quantile(.75, torch.rand((1, 2)))
    calc_quantile(.75, torch.rand((1, 3)))
    calc_quantile(.88, torch.rand((1, 4)))
    calc_quantile(.75, torch.rand((1, 5)))
    calc_quantile(.88, torch.tensor([[1., 2., 3., 4., 5., 6., 7.]]))
    calc_quantile(.88, torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]]))
    calc_quantile(.5, torch.rand((1, 100)))
    calc_quantile(.75, torch.rand((1, 10000)))
    calc_quantile(.43, torch.rand((1, 99999)))
    calc_quantile(.992, torch.rand((1, 1000000)))
    # batch size
    calc_quantile(.75, torch.rand((16, 5)))
    calc_quantile(.66, torch.rand((32, 512, 8, 8)))
    print('\n\n*** PASS ***')


if __name__ == '__main__':
    test_quantile()
