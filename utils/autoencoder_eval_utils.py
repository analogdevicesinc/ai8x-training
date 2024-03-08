###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
""" Some utility functions for AutoEncoder Models """
import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")


DECAY_FACTOR = 1


def calc_model_size(model):
    """
    Returns the model's weight anf bias number.
    """
    model.eval()
    num_weights = 0
    num_bias = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.endswith('weight'):
                num_weights += np.prod(param.size())
            elif name.endswith('bias'):
                num_bias += np.prod(param.size())

    print(f'\nNumber of Model Weights: {num_weights}')
    print(f'Number of Model Bias: {num_bias}\n')
    return num_weights, num_bias


def extract_reconstructions_losses(model, dataloader, device):
    """
    Calculates and returns reconstructed signal reconstruction loss, input signals
    and latent space representations for autoencoder model.
    """
    model.eval()
    loss_fn = nn.MSELoss(reduce=False)
    losses = []
    reconstructions = []
    inputs = []
    labels = []

    with torch.no_grad():
        for tup in dataloader:
            if len(tup) == 2:
                signal, label = tup
            elif len(tup) == 3:
                signal, label, _ = tup
            elif len(tup) == 4:
                signal, label, _, _ = tup

            signal = signal.to(device)
            label = label.type(torch.long).to(device)

            inputs.append(signal)
            labels.append(label)

            model_out = model(signal)
            if isinstance(model_out, tuple):
                model_out = model_out[0]

            loss = loss_fn(model_out, signal)
            loss_numpy = loss.cpu().detach().numpy()
            decay_vector = np.array([DECAY_FACTOR**i for i in range(loss_numpy.shape[2])])
            decay_vector = np.tile(decay_vector, (loss_numpy.shape[0], loss_numpy.shape[1], 1))

            decayed_loss = loss_numpy * decay_vector
            losses.extend(decayed_loss.mean(axis=(1, 2)))
            reconstructions.append(model_out)

    return reconstructions, losses, inputs, labels


def plot_all_metrics(F1s, BalancedAccuracies, FPRs, Recalls, percentiles):
    """
    F1, Balanced Accuracy, False Positive Rate metrics are plotted with respect to
    threshold decided according to percentiles of training loss in percentile list.
    """
    fontsize = 22
    linewidth = 4

    fig, axs = plt.subplots(1, 4, figsize=(36, 11))

    axs[0].plot(percentiles, F1s, '-o', linewidth=linewidth)
    for i, xy in enumerate(zip(percentiles, F1s)):  # pylint: disable=unused-variable
        axs[0].annotate(f"{xy[1]: .3f}", xy=xy, fontsize=fontsize - 2)

    axs[0].grid()

    axs[0].set_title('\nF1 Score on Testset\n\n', fontsize=fontsize + 4, color='#0070C0')
    axs[0].tick_params(axis='both', which='both', labelsize=fontsize)
    axs[0].legend(("F1 Score",), loc='lower left', fontsize=fontsize - 2)

    axs[1].plot(percentiles, BalancedAccuracies, '-o', linewidth=linewidth)
    for i, xy in enumerate(zip(percentiles, BalancedAccuracies)):
        axs[1].annotate(f"{xy[1]: .3f}", xy=xy, fontsize=fontsize - 2)

    axs[1].grid()

    axs[1].set_title('\nBalanced Accuracy ((TPRate + TNRate) / 2) on Testset\n\n',
                     fontsize=fontsize + 4, color='#0070C0')
    axs[1].tick_params(axis='both', which='both', labelsize=fontsize)
    axs[1].legend(("Balanced Acc.",), loc='lower left', fontsize=fontsize - 2)

    axs[2].plot(percentiles, FPRs, '-o', linewidth=linewidth)
    for i, xy in enumerate(zip(percentiles, FPRs)):
        axs[2].annotate(f"{xy[1]: .3f}", xy=xy, fontsize=fontsize - 2)

    axs[2].grid()
    axs[2].set_title('\nFalse Positive Rate on Testset\n\n',
                     fontsize=fontsize + 4, color='#0070C0')
    axs[2].tick_params(axis='both', which='both', labelsize=fontsize)
    axs[2].legend(("FPRate",), loc='lower left', fontsize=fontsize - 2)

    axs[3].plot(percentiles, Recalls, '-o', linewidth=linewidth)
    for i, xy in enumerate(zip(percentiles, Recalls)):
        axs[3].annotate(f"{xy[1]: .3f}", xy=xy, fontsize=fontsize - 2)

    axs[3].grid()
    axs[3].set_title('\nTrue Positive Rate on Testset\n\n', fontsize=fontsize + 4, color='#0070C0')
    axs[3].tick_params(axis='both', which='both', labelsize=fontsize)
    axs[3].legend(("Recall",), loc='lower left', fontsize=fontsize - 2)

    fig.supxlabel('\nReconstruction Loss distribution percentile of training samples (%)',
                  fontsize=fontsize + 4)

    plt.tight_layout()
    plt.show()


def sweep_performance_metrics(thresholds, train_tuple, test_tuple):
    """
    F1s, BalancedAccuracies, FPRs, Recalls are calculated
    and returned based on different thresholds.
    """

    train_reconstructions, train_losses, \
        train_inputs, train_labels = train_tuple  # pylint: disable=unused-variable
    test_reconstructions, test_losses, \
        test_inputs, test_labels = test_tuple  # pylint: disable=unused-variable

    FPRs = []
    F1s = []
    BalancedAccuracies = []
    Recalls = []

    for threshold in thresholds:
        FPRate, _, Recall, Precision, Accuracy, F1, BalancedAccuracy = calc_ae_perf_metrics(
            test_reconstructions,
            test_inputs,
            test_labels,
            threshold=threshold,
            print_all=False
            )

        _, _, _, _, AccuracyTrain, _, _ = calc_ae_perf_metrics(
            train_reconstructions,
            train_inputs,
            train_labels,
            threshold=threshold,
            print_all=False
            )

        F1s.append(F1.item())
        BalancedAccuracies.append(BalancedAccuracy.item())
        FPRs.append(FPRate.item())
        Recalls.append(Recall.item())

        print(f"F1: {F1: .4f}, BalancedAccuracy: {BalancedAccuracy: .4f}, "
              f"FPRate: {FPRate: .4f}, Precision: {Precision: .4f}, TPRate (Recall): "
              f"{Recall: .4f}, Accuracy: {Accuracy: .4f}, "
              f"TRAIN-SET Accuracy: {AccuracyTrain: .4f}")

    return F1s, BalancedAccuracies, FPRs, Recalls


def calc_ae_perf_metrics(reconstructions, inputs, labels, threshold, print_all=True):
    """
    False Positive Rate, TNRate, Recall, Precision, Accuracy, F1, BalancedAccuracy
    metrics of AutoEncoder are calculated and returned.
    """

    loss_fn = nn.MSELoss(reduce=False)
    FP = 0
    FN = 0
    TP = 0
    TN = 0

    Recall = -1
    Precision = -1
    Accuracy = -1
    F1 = -1
    FPRate = -1

    BalancedAccuracy = -1
    TNRate = -1   # specificity (SPC), selectivity

    for i, inputs_batch in enumerate(inputs):
        label_batch = labels[i]
        reconstructions_batch = reconstructions[i]
        # inputs_batch = inputs[i]

        loss = loss_fn(reconstructions_batch, inputs_batch)

        # Loss Decay
        loss_numpy = loss.cpu().detach().numpy()
        decay_vector = np.array([DECAY_FACTOR**i for i in range(loss_numpy.shape[2])])
        decay_vector = np.tile(decay_vector, (loss_numpy.shape[0], loss_numpy.shape[1], 1))
        decayed_loss = loss_numpy * decay_vector
        decayed_loss = torch.Tensor(decayed_loss).to(label_batch.device)

        loss_batch = decayed_loss.mean(dim=(1, 2))
        prediction_batch = loss_batch > threshold

        TN += torch.sum(torch.logical_and(torch.logical_not(prediction_batch),
                                          torch.squeeze(torch.logical_not(label_batch))))
        TP += torch.sum(torch.logical_and((prediction_batch),
                                          torch.squeeze(label_batch)))
        FN += torch.sum(torch.logical_and(torch.logical_not(prediction_batch),
                                          torch.squeeze(label_batch)))
        FP += torch.sum(torch.logical_and((prediction_batch),
                                          torch.squeeze(torch.logical_not(label_batch))))

    if TP + FN != 0:
        Recall = TP / (TP + FN)

    if TP + FP != 0:
        Precision = TP / (TP + FP)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    if (TN + FP) != 0:
        FPRate = FP / (TN + FP)
        TNRate = TN / (TN + FP)

    if Precision + Recall != 0:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)

    BalancedAccuracy = (Recall + TNRate) / 2

    if print_all:
        print(f"TP: {TP}")
        print(f"FP: {FP}")
        print(f"TN: {TN}")
        print(f"FN: {FN}")
        print(f"FPRate: {FPRate}")
        print(f"TNRate = Specificity: {TNRate}")
        print(f"TPRate (Recall): {Recall}")
        print(f"Precision: {Precision}")
        print(f"Accuracy: {Accuracy}")
        print(f"F1: {F1}")
        print(f"BalancedAccuracy: {BalancedAccuracy}")

    return FPRate, TNRate, Recall, Precision, Accuracy, F1, BalancedAccuracy
