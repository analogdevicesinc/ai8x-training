###################################################################################################
# Copyright (C) 2020-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Part number and device type conversion
"""
import argparse


def device(astring):
    """
    Take die type, or part number, and return the die type.
    """
    s = astring.lower()

    if s.startswith('max'):
        s = s[3:]  # Strip 'MAX' from part number
    elif s.startswith('ai'):
        s = s[2:]  # Strip 'AI' from die type

    try:
        num = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(astring, 'is not a supported device type') from exc
    if num in [84, 85, 87]:  # Die types
        dev = num
    elif num == 78000:  # Part numbers
        dev = 85
    elif num == 78002:
        dev = 87
    else:
        raise argparse.ArgumentTypeError(astring, 'is not a supported device type')

    return dev


def partnum(num):
    """
    Return part number for a die type.
    """
    if num == 84:
        return 'AI84'
    if num == 85:
        return 'MAX78000'
    if num == 87:
        return 'MAX78002'

    raise RuntimeError(f'Unknown die type {num}')
