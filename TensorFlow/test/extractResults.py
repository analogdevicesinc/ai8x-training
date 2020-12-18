###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
""" extract final output of tests and store in a log
"""
import argparse
import os
import random as rn
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

# following piece it to init seed to make reproducible results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
rn.seed(100)
tf.random.set_seed(7)

VERBOSE = 2

# command parser
parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '--folder',
    action='store',
    dest='folder',
    type=str,
    default='./',
    help='the folder including all test folders')
parser.add_argument(
    '--result',
    action='store',
    dest='result',
    type=str,
    default='saved_model/result.log',
    help='reseult file name')
parser.add_argument(
    '--keyword',
    action='store',
    dest='keyword',
    type=str,
    default='Output(8',
    help='keyword in result file to grap for final output')
parser.add_argument(
    '--skip-exec',
    action='store_true',
    dest='skip',
    default=False,
    help='Skip running shell scripts and just extract output log')
args = parser.parse_args()


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.realpath(__file__))

    folder = args.folder
    result = args.result
    keyword = args.keyword
    skip = args.skip
    print(keyword)
    print(folder)

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    output = open('test_result_' + date_time + '.log', 'w+')

    output.writelines(f'Test started at: {date_time}\n')
    count = 0
    for dirpath, dirnames, files in os.walk(os.path.join(folder)):
        # print(f'{dirpath}, {dirnames}, {files}')
        if dirpath == './':
            continue
        for file in files:
            if file.endswith('.sh'):
                os.chdir(dirpath)

                if not skip:
                    shell = 'bash ' + file
                    print(f'executing {dirpath}/{shell}')

                    # remove old result
                    os.system('rm -R saved_model')
                    os.system('chmod 777 *.sh')

                    # run new test
                    os.system(shell)

                # find output
                logfile = os.path.join(result)
                count += 1
                # print(logfile)
                try:
                    with open(logfile) as myfile:
                        print(f'checking result of {dirpath}')
                        output.writelines('-' * 50)
                        output.writelines(f'\nTest({count}): {dirpath}\n')
                        notfound = True
                        for line in myfile:
                            if not line.startswith(keyword) and notfound:
                                continue
                            notfound = False
                            print('>>>'+line.rstrip())
                            output.writelines(line.rstrip() + '\n')
                except:  # pylint: disable=W0702 # noqa
                    print('file not found')
                    output.writelines(f'\n{logfile} not found in {dirpath}\n')

                os.chdir(working_path)
    output.close()
    sys.exit(0)
