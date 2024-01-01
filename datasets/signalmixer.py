from mixedkws import MixedKWS
import mixedkws

import numpy as np
import torch
import os

import ai8x

class signalmixer:

    def __init__(self, signal_dataset, noise = False, noise_dataset = None, snr = None, noise_kind = None, 
                    quantized_noise = False, transform = None):

        self.signal_data = signal_dataset.data
        self.signal_targets = signal_dataset.targets

        self.noise = noise

        if (self.noise and noise_kind != 'WhiteNoise'):
            self.noise_data = noise_dataset.data
            self.noise_targets = noise_dataset.targets
            
            if quantized_noise:
                self.noise_dataset_float = next(iter(torch.utils.data.DataLoader(noise_dataset, batch_size = noise_dataset.len)))[0]
            else:
                self.noise_dataset_float = next(iter(torch.utils.data.DataLoader(noise_dataset, batch_size = noise_dataset.len)))[0]
                
            self.noise_rms = noise_dataset.rms

        self.snr = snr
        self.quantized_noise = quantized_noise
        self.transform = transform
        self.noise_kind = noise_kind

        # using getitem to reach the test data ()
        self.test_dataset_float = next(iter(torch.utils.data.DataLoader(signal_dataset, batch_size = signal_dataset.data.shape[0])))[0]

        if noise:
            if (noise_kind == 'WhiteNoise'):
                self.mixed_signal = self.white_noise_mixer()
            else:
                self.mixed_signal = self.snr_mixer()
        
    def __getitem__(self, index):

        inp, target = self.mixed_signal[index].type(torch.FloatTensor), int(self.signal_targets[index])
        return inp, target

    def __len__(self):
        return len(self.mixed_signal)

    def snr_mixer(self):

        clean = self.test_dataset_float
        noise = self.noise_dataset_float

        idx = np.random.randint(0, noise.shape[0], clean.shape[0])
        noise = noise[idx]
        rms_noise = self.noise_rms[idx]

        snr = self.snr

        rmsclean = (torch.mean(clean.reshape(clean.shape[0], -1)**2, 1, keepdims = True)**0.5).unsqueeze(1)
        scalarclean = 1 / rmsclean
        clean = clean * scalarclean

        scalarnoise = 1 / rms_noise.reshape(-1,1,1)
        noise = noise * scalarnoise

        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor*clean + noise
        noisyspeech = noisyspeech / (torch.tensor(scalarnoise) + cleanfactor * scalarclean)

        # 16384 --> (noisyspeech[0].shape[0])*(noisyspeech[0].shape[1])
        max_mixed = torch.max(abs(noisyspeech.reshape(noisyspeech.shape[0], (noisyspeech[0].shape[0])*(noisyspeech[0].shape[1]))), 1, keepdims = True).values
        # noisyspeech = noisyspeech * (torch.where(max_mixed != 0, 1.0 / max_mixed, max_mixed)).unsqueeze(1)

        noisyspeech = noisyspeech * (1/max_mixed).unsqueeze(1)
        return noisyspeech
    
    def white_noise_mixer(self):

        clean = self.test_dataset_float
        snr = self.snr

        mean = 0
        std = 1
        noise = np.random.normal(mean, std, clean.shape)
        noise = torch.tensor(noise, dtype = torch.float32)

        rmsclean = (torch.mean(clean.reshape(clean.shape[0], -1)**2, 1, keepdims = True)**0.5).unsqueeze(1)
        scalarclean = 1 / rmsclean
        clean = clean * scalarclean

        rmsnoise = (torch.mean(noise.reshape(noise.shape[0], -1)**2, 1, keepdims = True)**0.5).unsqueeze(1)
        scalarnoise = 1 / rmsnoise
        noise = noise * scalarnoise

        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor*clean + noise
        noisyspeech = noisyspeech / (scalarnoise + cleanfactor * scalarclean)

        max_mixed = torch.max(abs(noisyspeech.reshape(noisyspeech.shape[0], 16384)), 1, keepdims = True).values
        noisyspeech = noisyspeech * (torch.where(max_mixed != 0, 1.0 / max_mixed, max_mixed)).unsqueeze(1)

        return noisyspeech
