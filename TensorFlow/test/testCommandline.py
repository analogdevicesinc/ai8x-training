###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""train a keras sequential model
"""
from datetime import datetime
from pydoc import locate
import sys
import random as rn
from random import randint
import os
import fnmatch
import argparse
import tensorflow as tf
import numpy as np

# following piece it to init seed to make reproducable results
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
# parser.print_help()
# print('input args: ', args)


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.realpath(__file__))

    folder = args.folder
    result = args.result
    keyword = args.keyword
    skip = args.skip
    print(keyword)
    print(folder)
    datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    output = open('test_result' + datetime + '.log', 'w+')
    
    output.writelines(f'\nTest time: {datetime}\n')
    count = 0
    for dirpath , dirnames , files in os.walk ( os.path.join(folder) ):
        #print(f'{dirpath}, {dirnames}, {files}')
        if dirpath == './':
            continue
        for file in files:
            if file.endswith ( '.sh' ) :
                os.chdir(dirpath)
                
                
                if not skip:
                    shell = 'bash ' + file # os.path.join(dirpath, file)
                    print(f'executing {dirpath}/{shell}')
                    
                    # remove old result
                    os.system('rm -R saved_model')
                    os.system('chmod 777 *.sh')
                   
                    # run new test
                    #os.system('python test.py')
                    #os.system('python -m tf2onnx.convert --saved-model saved_model --inputs-as-nchw input_1:0 --opset 10 --output saved_model/saved_model.onnx')
                    os.system(shell)

                # find output
                logfile = os.path.join(result) #'saved_model','result.log')
                count += 1
                #print(logfile)
                try:
                    with open(logfile) as myfile:  # Open lorem.txt for reading text
                        #contents = myfile.read()  # Read the entire file to a string
                        print(f'checking result of {dirpath}')
                        output.writelines('-' * 50)
                        output.writelines(f'\nTest({count}): {dirpath}\n')
                        notfound = True
                        for line in myfile:
                           # print(line.rstrip())
                            if not line.startswith(keyword) and notfound:
                                continue
                            #if(not line.find('bit')):
                            #    continue
                            notfound = False 
                            print('>>>'+line.rstrip())
                            output.writelines(line.rstrip() + '\n')
                except:
                    print('file not found')
                    output.writelines(f'\n{logfile} not found in {dirpath}\n')
                    
                        
                os.chdir(working_path)
                #exit(0)
    output.close()
    exit(0)
