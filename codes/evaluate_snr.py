import os
import sys
import time

import numpy as np
import torch
import pandas as pd
import librosa

import matplotlib.pyplot as plt

import torchnet.meter as tnt
from collections import OrderedDict
import importlib
from torchvision import transforms

sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), './models/'))
sys.path.append(os.path.join(os.getcwd(), './datasets/'))

from types import SimpleNamespace

import ai8x

from msnoise import MSnoise
import msnoise
from signalmixer import signalmixer

from IPython.display import clear_output
from scipy.io.wavfile import write
import IPython

dataset = importlib.import_module("kws20")

classes = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
           'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'unknown']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Working with device:", device)

ai8x.set_device(device=85, simulate=False, round_avg=False)
qat_policy = {'start_epoch': 10, 'weight_bits': 8, 'bias_bits': 8}

# for KWS test_dataset_pth -> /data_ssd/
# for MSnoise noise_dataset_pth -> /data_ssd/

def evaluate_snr(checkpoint_pth, model_file, test_dataset_pth, noise_dataset_pth, noise_list, snr_list, saving_path):

    mod = importlib.import_module(model_file)
                                  
    if model_file == "ai85net-kws20-v2":

        model = mod.AI85KWS20Netv2(num_classes=len(classes), num_channels=128, dimensions=(128, 1), bias=False, 
                                quantize_activation=False)

    elif model_file == "ai85net-kws20-v3":

        model = mod.AI85KWS20Netv3(num_classes=len(classes), num_channels=128, dimensions=(128, 1), bias=False, 
                                quantize_activation=False)

    elif model_file == "ai85nasnet_kws20_res_1":

        model = mod.AI85NASNET_KWS20_RES_1(num_classes=len(classes), num_channels=128, dimensions=(128, 1), bias=True, 
                                quantize_activation=False)

    checkpoint = torch.load(checkpoint_pth)

    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['state_dict'] = new_state_dict

    ai8x.fuse_bn_layers(model)
    ai8x.initiate_qat(model, qat_policy)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    ai8x.update_model(model)

    # args
    sn = SimpleNamespace()
    sn.truncate_testset = False
    sn.act_mode_8bit = False

    _, test_dataset = dataset.KWS_20_get_datasets((test_dataset_pth, sn), load_train=False, load_test=True)
    
    # taking non-augmented test data
    originals = list(range(0, len(test_dataset), 3))
    test_dataset.data = test_dataset.data[originals]
    test_dataset.targets = test_dataset.targets[originals]

    # evaluation
    accuracies = benchmark(model, noise_list)
    
    # accuracy plots for per-model
    for idx, i in enumerate(noise_list):
        csv[i] = accuracies[idx]
        
    csv_list = []

    csv_list.append(['raw', 'None', accuracies_w[-1][0]])

    for i in csv.keys():
        for idx, j in enumerate(csv[i]):
            csv_list.append([i, snr[idx], j])

    df = pd.DataFrame(csv_list, columns = ['Type', 'SNR (dB)', f'{model_file}'])

    # csv writing
    if os.path.exists(os.path.join(saving_path, '/acc1.csv')):
        current = pd.read_csv(os.path.join(saving_path, '/acc1.csv'), decimal=',', sep=';')
        if model_file in current.columns:
            print(f'This model file ({model_file}) already exists!')
        else:
            current[model_file] = list(df[model_file].values)
            current.to_csv(os.path.join(saving_path, '/acc1.csv'), sep=';', decimal=',', index=False)

    else:
        df.to_csv(os.path.join(saving_path, '/acc1.csv'), sep=';', decimal=',', index=False)


def evaluate(model, db = None, noise = False, noise_kind = None):

    if (noise == True) and (noise_kind == None):
        print('Noise kind is not specified. Noise will not be applied.')
        noise = False

    model.eval()
    model.to(device)
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(len(classes), 5)))
    outputs_all = np.zeros((len(test_dataset), 21))
    targets_all = np.zeros((len(test_dataset)))

    transform = transforms.Compose([
        ai8x.normalize(args=sn)
    ])
    
    if noise:
        if (noise_kind == 'WhiteNoise'):

            mixed_signals = signalmixer(test_dataset, noise = True, noise_dataset = None, snr = db, noise_kind = noise_kind, 
                                        quantized_noise = False, transform = None)
            mixed_signals_loader = torch.utils.data.DataLoader(mixed_signals, batch_size = 256)

        else:
            noise_dataset = msnoise.MSnoise(root = noise_dataset_pth, classes = [noise_kind], d_type = 'test', remove_unknowns=True,
                        transform=None, quantize=False, download=False)

            mixed_signals = signalmixer(test_dataset, noise = True, noise_dataset = noise_dataset, snr = db, noise_kind = noise_kind, 
                    quantized_noise = False, transform = None)
            
            mixed_signals_loader = torch.utils.data.DataLoader(mixed_signals, batch_size = 256)
    else:
        mixed_signals_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256)
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(mixed_signals_loader):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            classerr.add(outputs, targets)

            print("Batch: [",batch_idx*256 ,"/", len(test_dataset),"]")
            acc = classerr.value()[0]
            print("Accuracy: ", acc)
            outputs_all[batch_idx*256:batch_idx*256+len(inputs)] = outputs.cpu().numpy()
            targets_all[batch_idx*256:batch_idx*256+len(inputs)] = targets.cpu().numpy()
            
    print("Total Accuracy: ", acc)
    return acc

def snr_testing(model, snr_list = None, noise_kind = None, noise = False):

    # raw test set evaluation
    if noise == False:
        db = None
        noise_kind = None

        accuracies = np.zeros(1)
        accuracies[0] = evaluate(model, db, noise, noise_kind)
    
        return accuracies

    snr_num = calculate_snr_num(snr_list)

    # noisy test set evaluation
    accuracies = np.zeros(snr_num)

    for idx, db in enumerate(snr_list):
        print("Evaluating SNR levels of", db)
        accuracies[idx] = evaluate(model, db, noise, noise_kind)

    return accuracies

def benchmark(model, noise_list = None, snr_list = None):

    if noise_list == None:
        print('Noise kind is not specified. Noise will not be applied.')
        noise = False
        snr_list = None
    else:
        noise = True
        if snr_list == None:
            print('Using default values for SNR levels: [-5, 20] dB.')
            snr_list = range(-5, 20)

    if noise:

        snr_num = calculate_snr_num(snr_list)
        num = len(noise_list) + 1

        accuracies = np.zeros((num, snr_num))

        for idx, n in enumerate(noise_list):
            print(f'{n} Noise Evaluation')
            accuracies[idx] = snr_testing(model, snr_list, noise_kind = n, noise = noise)

    accuracies[-1] = snr_testing(model, noise = noise)

    return accuracies   

def calculate_snr_num(snr_list):
    snr_num = len(snr_list)
    return snr_num


#plotting

def plot_models(saving_path, snr_list):

    data_files = pd.read_csv(os.path.join(saving_path, '/acc1.csv'), decimal=',', sep=';')

    for model_num in data_files.columns[2:]:
        accuracies_values = plot_values(model_num)
        plt.figure()
        plt.grid()
        for noise in noise_list:
            plt.title(f'Accuracy Test on model {model_num}') #duzelecek
            plt.xlabel('SNR (dB)')
            plt.ylabel('Accuracy (%)')
            plt.plot(snr_list, accuracies_values[noise])

        plt.legend(noise_list, bbox_to_anchor=(1.05, 0.75),
                            loc='upper left', borderaxespad=0.)

def plot_noises(saving_path, snr_list):

    data_files = pd.read_csv(os.path.join(saving_path, '/acc1.csv'), decimal=',', sep=';')

    acc_list = []
    for model_num in data_files.columns[2:]:
        acc_list.append(plot(model_num))
        
    for noise in noise_list:
        plt.figure()
        plt.grid()
        plt.title(f'{noise} - Accuracy Test')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy (%)')
        for model in acc_list:
            plt.plot(snr , model[noise])

        plt.legend(data_files.columns[2:], bbox_to_anchor=(1.05, 0.75),
                            loc='upper left', borderaxespad=0.)

def plot_values(model_num):

    accuracies_values = {}

    for noise in noise_list:
        acc_list = []
        for idx, i in enumerate(data_files['Type'].values[1:]): 
            if i == noise:
                acc_list.append(data_files[model_num][1:][idx+1])
        accuracies_values[noise] = acc_list

    return accuracies_values



evaluate_snr(checkpoint_pth = '/home/merveeyuboglu/Github/ai8x-training/codes/checkpoints/TrueVAL_NAS_0911Speed_DynAug/qat_best.pth.tar',
                model_file = 'ai85nasnet_kws20_res_1', 
                test_dataset_pth = '/data_ssd/', 
                noise_dataset_pth = '/data_ssd/', 
                noise_list = ['Typing', 'Babble'], 
                snr_list = range(-5,31), 
                saving_path = '/home/merveeyuboglu/Github/ai8x-training/codes')

#plot_models(saving_path = '/home/merveeyuboglu/Github/ai8x-training/codes', snr_list = range(-5,31))