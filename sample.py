###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Generate a sample for KAT
"""
import numpy as np


def generate(
        index,
        inputs,
        targets,  # pylint: disable=unused-argument
        outputs,  # pylint: disable=unused-argument
        dataset_name,
        search=False,  # pylint: disable=unused-argument
        slice_sample=False,
):
    """
    Save the element `index` from the `inputs` batch to a file named "sample_`dataset_name`.npy".
    If `search`, then check `outputs` against `targets` until a match is found.
    """
    if index >= len(inputs):
        raise ValueError('--generate-sample index is larger than the data batch size')

    sample_name = 'sample_' + dataset_name.lower()

    # FIXME: Implement search

    print(f'==> Saving sample at index {index} to {sample_name}.npy')
    x = inputs[index].cpu().numpy().astype('int64')
    if slice_sample:
        x = x[0:3, :, :]
    x = np.clip(x, -128, 127)
    np.save(sample_name, x, allow_pickle=False, fix_imports=False)

    return False
