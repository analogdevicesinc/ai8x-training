###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Override Distiller's input shape inference.
"""
dimensions = None


def get(
        dataset,  # pylint: disable=unused-argument
        input_shape,  # pylint: disable=unused-argument
):
    """
    Return the input shape of our selected dataset. Ignores all the built-in functionality,
    and the arguments are not used either. Dimensions are set up in main().
    """
    return (1, ) + dimensions  # Batch dimension plus configured shape
