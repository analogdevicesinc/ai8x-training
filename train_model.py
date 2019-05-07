#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Train various models
"""
import argparse
import datetime
import logging
import os
import signal
import sys
import time
from pydoc import locate
import numpy as np
import progressbar
import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter  # 1.1: from torch.utils.tensorboard import SummaryWriter
from python_utils import terminal
sys.path.append('distiller')
# pylint: disable=wrong-import-position
import distiller  # noqa E402
# pylint: enable=wrong-import-position


DEFAULT_LR = 0.01
DEFAULT_LEGACY_WEIGHT_DECAY = 0.001  # Legacy checkpoints that didn't store this


def main():
    """
    Main training program
    """
    term_width, _ = terminal.get_terminal_size()
    np.set_printoptions(edgeitems=5, linewidth=term_width)

    best_loss = float('inf')
    best_accuracy = 0
    start_epoch = 0
    optimizer = None
    weight_decay = 0.
    adjusting_weight_decay = False
    adjust_weight_decay_hysteresis = 0
    best_epoch = -1
    last_saved_epoch = -1

    # https://pytorch.org/docs/master/optim.html
    supported_optimizers = [
        {'name': 'Adadelta',
         'weight_decay': True,
         'centered': False,
         'momentum': False},
        {'name': 'Adagrad',
         'weight_decay': True,
         'centered': False,
         'momentum': False},
        {'name': 'Adam',
         'weight_decay': True,
         'centered': False,
         'momentum': False},
        {'name': 'Adamax',
         'weight_decay': True,
         'centered': False,
         'momentum': False},
        {'name': 'ASGD',
         'weight_decay': True,
         'centered': False,
         'momentum': False},
        {'name': 'RMSprop',
         'weight_decay': True,
         'centered': True,
         'momentum': True},
        {'name': 'Rprop',
         'weight_decay': False,
         'centered': False,
         'momentum': False},
        {'name': 'SGD',
         'weight_decay': True,
         'centered': False,
         'momentum': True}
    ]

    # https://pytorch.org/docs/master/nn.html?highlight=nllloss#loss-functions
    supported_loss_functions = [
        'L1',
        'MSE',
        'SmoothL1',
        'KLDiv',
        'NLL',
        'CrossEntropy',
        'MultiMargin'
    ]

    # https://pytorch.org/docs/master/torchvision/datasets.html
    supported_sources = [
        {'name': 'MNIST',
         'input': (1, 28, 28),
         'output': list(map(str, range(10)))},
        {'name': 'FashionMNIST',
         'input': (1, 28, 28),
         'output': ('top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                    'shirt', 'sneaker', 'bag', 'boot')},
        {'name': 'CIFAR10',
         'input': (3, 32, 32),
         'output': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                    'ship', 'truck')}
    ]

    supported_image_types = [
        '2d'
    ]

    supported_models = [
        {'name': 'resnet10',  # Model name (command line, class name)
         'module': 'resnet',  # Python module
         'min_input': 1,      # Minimum size (H/W) of input (with default kernel, padding, stride)
         'dim': 2},           # Dimensions (1=1D, 2=2D)
        {'name': 'resnet18',
         'module': 'resnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'resnet34',
         'module': 'resnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'resnet50',
         'module': 'resnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'resnet101',
         'module': 'resnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'sresnet4',
         'module': 'sresnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'rsresnet4',
         'module': 'sresnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'sresnet6',
         'module': 'sresnet',
         'min_input': 1,
         'dim': 2},
        {'name': 'sresnet8',
         'module': 'sresnet',
         'min_input': 1,
         'dim': 2}
    ]

    class TerminalHelpFormatter(argparse.HelpFormatter):
        """
        HelpFormatter that respects the window width
        """
        def __init__(self, **kwargs):
            super(TerminalHelpFormatter, self).__init__(width=term_width,
                                                        max_help_position=34,
                                                        **kwargs)

    optimizers = [item['name'] for item in supported_optimizers]
    optimizers_momentum = [item['name'] for item in supported_optimizers if item['momentum']]
    optimizers_weight_decay = [item['name'] for item in
                               supported_optimizers if item['weight_decay']]
    cnn_models = [item['name'] for item in supported_models]
    datasets = [item['name'] for item in supported_sources]

    parser = argparse.ArgumentParser(description="AI84 Deep Neural Network Model Training",
                                     formatter_class=TerminalHelpFormatter)
    parser.add_argument('--accuracy', type=float, default=5.0, metavar='F',
                        help="maximum delta to be considered correct (mmHg, default: 5.0)")
    parser.add_argument('--add-logsoftmax', action='store_true', default=False,
                        help="add a final LogSoftmax() operation to model (default: disabled)")
    parser.add_argument('--add-softmax', action='store_true', default=False,
                        help="add a final Softmax() operation to model (default: disabled)")
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help="input batch size for training (default: 32)")
    # parser.add_argument('--batch-accumulate', type=int, default=1, metavar='N',
    #                     help="accumulate multiple batches using gradient accumulation "
    #                          "(default: 1)")
    parser.add_argument('--eval-batch-size', type=int, default=64, metavar='N',
                        help="input batch size for evaluation (default: 64)")
    parser.add_argument('--compress', metavar='S',
                        help="compress/prune the model as specified in configuration file")
    parser.add_argument('--cnn', '--model', metavar='S', default='resnet18',
                        choices=cnn_models,
                        help="CNN model (" + ', '.join(cnn_models) +
                        "; default: resnet18)")
    parser.add_argument('--checkpoint', default='', metavar='S',
                        help="add string to checkpoint name (default: empty)")
    parser.add_argument('--checkpoint-directory', default='checkpoint', metavar='S',
                        help="checkpoint directory")
    parser.add_argument('--checkpoint-always', action='store_true', default=False,
                        help="shortcut for --checkpoint-save-frequency 1")
    parser.add_argument('--checkpoint-save-frequency', type=int, metavar='N',
                        help="save state every N epochs (default: disabled)")
    parser.add_argument('--checkpoint-on-quit', action='store_true', default=False,
                        help="save state on last epoch (default: disabled)")
    parser.add_argument('--dataset', default=['FashionMNIST'], metavar='S',
                        choices=datasets, help="dataset(s) (" + ', '.join(datasets) +
                        "; default: FashionMNIST)")
    parser.add_argument('--cv-dataset', metavar='S', nargs='+',
                        choices=datasets,
                        help="cross validation dataset(s) (option defaults to --dataset if not "
                             "specified)")
    parser.add_argument('--test-dataset', metavar='S', nargs='+',
                        choices=datasets, help="evaluate test dataset(s) (none if not specified)")
    parser.add_argument('--test-dataset2', metavar='S', nargs='+',
                        choices=datasets,
                        help="evaluate second test dataset(s) (none if not specified)")
    parser.add_argument('--disable-checkpoints', action='store_false', default=True,
                        dest='checkpoints',
                        help="disable generating and saving checkpoints (default: enabled)")
    # parser.add_argument('--disable-tensorboard', action='store_false', default=True,
    #                     dest='tb_log',
    #                     help="disable Tensorboard logging (default: enabled)")
    parser.add_argument('-e', '--epochs', type=int, default=20, metavar='N',
                        help="number of epochs to train (default: 20)")
    parser.add_argument('--gpu', type=int, metavar='N',
                        help="GPU number to train on (default: all)")
    parser.add_argument('--input-type', metavar='S',
                        choices=supported_image_types, default='2d',
                        help="input image type (" + ', '.join(supported_image_types) +
                        "; default: 2d)")
    parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                        help="minimum number of seconds to wait before logging training status"
                        " (default: 3)")
    parser.add_argument('--log-level', type=int, default=logging.INFO, metavar='N',
                        choices=range(51),
                        help="log level (0 to 50, defaults to 20 or INFO)")
    parser.add_argument('--loss-function', metavar='S',
                        choices=supported_loss_functions,
                        help="loss function (" + ', '.join(supported_loss_functions) +
                        "; default: MSE for regression, CrossEntropy for classification)")
    parser.add_argument('--loss-threshold', type=float, metavar='F',
                        help="consider all losses smaller than this value to be zero "
                             "(default: no limit)")
    parser.add_argument('--lr', '--learning-rate', type=float, metavar='F',
                        help=f"initial learning rate (default: {DEFAULT_LR}, "
                             f"or previous on --resume)")
    parser.add_argument('--lr-limit', type=float, metavar='F',
                        help=f"terminate when learning rate reaches threshold (default: disabled)")
    parser.add_argument('--lr-min', type=float, metavar='F',
                        help=f"do not decrease learning rate further than this value "
                             f"(default: no minimum)")
    parser.add_argument('--lr-patience', type=int, default=10, metavar='N',
                        help="learning rate scheduler patience (default: 10)")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='F',
                        help="momentum (for " + ', '.join(optimizers_momentum) + "; default: 0.9)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help="disable CUDA training")
    parser.add_argument('--no-cudnn', action='store_true', default=False,
                        help="disable cuDNN backend (default: enabled)")
    parser.add_argument('--no-parallel', action='store_true', default=False,
                        help="disable parallel training")
    parser.add_argument('--no-stratify', action='store_true', default=False,
                        help="disable class stratification via WeightedRandomSampler")
    parser.add_argument('--onnx', metavar='S',
                        help="export ONNX model to file")
    parser.add_argument('--optimizer', metavar='S', default='Adam',
                        choices=optimizers,
                        help="optimizer (" + ', '.join(optimizers) + "; default: Adam)")
    parser.add_argument('--infer-without-truth', action='store_true', default=False,
                        help="evaluate samples even when no ground truth available")
    parser.add_argument('--print-model', action='store_true', default=False,
                        help="print model")
    parser.add_argument('--resume', '--resume-last', '-r', action='store_true', default=False,
                        dest='resume', help="resume from last checkpoint")
    parser.add_argument('--resume-best', action='store_true', default=False,
                        help="resume from best checkpoint")
    parser.add_argument('--seed', type=int, metavar='N',
                        help="random seed")
    parser.add_argument('--stdout', action='store_true',
                        help="force stdout redirection-friendly output (default: auto detect)")
    parser.add_argument('--summary-log', metavar='S',
                        help="save summary log in CSV form to file")
    parser.add_argument('--test', action='store_true', default=False,
                        help="test model against test set")
    parser.add_argument('--trace', action='store_true', default=False,
                        help="trace input and output sizes though network")
    parser.add_argument('--train-with-testset', action='store_true', default=False,
                        help="include validation and test sets in training (default: false)")
    parser.add_argument('--truncate', type=int, metavar='N',
                        help="truncate training set (debug)")
    parser.add_argument('--use-amp', dest='no_amp', action='store_false', default=True,
                        help="enable CUDA Apex Amp mixed precision acceleration")
    parser.add_argument('--weight-decay', type=float, metavar='F',
                        help="initial weight decay for " + ', '.join(optimizers_weight_decay) +
                        " (default: 0, or previous on --resume)")
    parser.add_argument('--weight-decay-threshold', type=float, metavar='F',
                        help="adjust weight decay up/down when threshold loss is crossed")
    parser.add_argument('--weight-decay-min', type=float, metavar='F', default=0.001,
                        help="smallest weight decay value when adjusting (default: 0.001)")
    parser.add_argument('--weight-decay-max', type=float, metavar='F', default=0.1,
                        help="largest weight decay value when adjusting (default: 0.1)")
    parser.add_argument('--weight-decay-momentum', type=int, metavar='N', default=2,
                        help="factor for weight decay adjustment (default: 2)")
    parser.add_argument('--weight-decay-hysteresis', type=int, metavar='N', default=2,
                        help="delay in epochs before adjusting weight decay again")
    parser.add_argument('--workers', type=int, default=8, metavar='N',
                        help="CPU workers (default: 8), ignored when using CUDA")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.resume_best:
        args.resume = True
    if not args.stdout:
        use_stdout = not sys.stdout.isatty()
    else:
        use_stdout = True
    if args.loss_function is None:
        args.loss_function = 'CrossEntropy'
    if args.lr and args.lr > 1.0:
        parser.error("--learning-rate must be 1.0 or lower")
    if args.summary_log and args.epochs == 0:
        parser.error("--summary-log requires --epochs larger than 0")
    if args.checkpoint_always:
        if args.checkpoint_save_frequency is not None:
            parser.error("cannot combine --checkpoint-save-frequency and --checkpoint-always")
        args.checkpoint_save_frequency = 1
    if args.cv_dataset is None:
        # Default to training set
        args.cv_dataset = args.dataset

    if args.test:
        args.resume = True
        args.epochs = 0

    # Fix this later -- logs are currently disabled
    args.tb_log = False

    print(f"Command line: {' '.join(str(x) for x in sys.argv)}")

    logging.basicConfig(level=args.log_level, format='%(message)s')

    print(f"PyTorch {torch.__version__}; ", end='')
    print(f"Random seeds: {np.random.get_state()[1][0]}, {torch.initial_seed()}; ", end='')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        ngpu = torch.cuda.device_count()
        if args.gpu is not None:
            print(f"CUDA enabled, using {ngpu} GPU{'s' if ngpu > 1 else ''}, #{args.gpu}")
        else:
            print(f"CUDA enabled, using {ngpu} GPU{'s' if ngpu > 1 else ''}")
        args.workers = 1
        dev = 'cuda'

        import apex  # pylint: disable=import-error, unused-import  # noqa: F401
    else:
        ngpu = 0
        print(f"Using CPU, {args.workers} workers")
        dev = 'cpu'
        if args.gpu is not None:
            parser.error("--gpu requested, but CUDA disabled or not found")
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    device = torch.device(dev)

    if args.dataset == 'CIFAR10':
        augment = [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip()]
    else:
        augment = [transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5)]

    kwargs = {'pin_memory': True} if use_cuda else {}

    module = next(item for item in supported_models if item['name'] == args.cnn)

    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    labels = selected_source['output']
    num_classes = len(labels)
    dimensions = selected_source['input']

    DataSet = locate('torchvision.datasets.' + args.dataset)
    if args.dataset == 'CIFAR10':
        data_root = os.path.join('data', args.dataset)
    else:
        data_root = 'data'

    if args.epochs != 0:
        train_set = torch.utils.data.ConcatDataset([
            DataSet(data_root, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ])),
            DataSet(data_root, train=True, download=True,
                    transform=transforms.Compose(  # Data augmentation
                        augment +
                        [transforms.ToTensor()]
                    ))])

        # def balanced_weights(ds):
        #     count = num_classes
        #     with np.errstate(divide='ignore'):  # This sets div 0 to "inf" which is fine
        #         class_weights = count.sum() / count
        #     weights = np.zeros(len(ds.samples), dtype=np.float64)
        #     for idx, val in enumerate(ds.samples):
        #         # Calculate indirectly to support pdf
        #         c = val[2]
        #         weights[idx] = class_weights[c]
        #     return weights

        # To stratify weight classes for unbalanced datasets, create a weighted sampler
        # train_sampler = torch.utils.data.sampler.\
        #     WeightedRandomSampler(balanced_weights(train_set), len(train_set)) \
        #     if not args.no_stratify else None

        train_loader = torch.utils.data.DataLoader(
            train_set,  # sampler=train_sampler,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, **kwargs)

        train_len = len(train_loader.dataset)
    else:
        train_len = 0

    eval_set = DataSet(data_root, train=False,
                       transform=transforms.Compose([transforms.ToTensor()]))

    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.eval_batch_size,
                                              shuffle=True, num_workers=args.workers, **kwargs)

    # Optionally, also create a test dataset. This is not something we should usually do.
    # if args.test_dataset:
    #     test_set = bpds.Dataset(args.source, validate=False, test=True,
    #                             classes=num_classes,
    #                             add_no_truth=False,
    #                             dimensions=module['dim'],
    #                             ds=args.test_dataset)

    #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size,
    #                                               shuffle=True, num_workers=args.workers,
    #                                               **kwargs)

    # # And a second test dataset.
    # if args.test_dataset2:
    #     test_set2 = bpds.Dataset(args.source, validate=False, test=True,
    #                              classes=num_classes,
    #                              add_no_truth=False,
    #                              dimensions=module['dim'],
    #                              ds=args.test_dataset2)

    #     test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=args.eval_batch_size,
    #                                                shuffle=True, num_workers=args.workers,
    #                                                **kwargs)

    input_type_name = args.input_type
    print(f"Training: {train_len} ({args.dataset}), evaluation: "
          f"{len(eval_loader.dataset)} ({args.cv_dataset}), ", end='')
    # if args.test_dataset:
    #     print(f"test: {len(test_loader.dataset)} ({'/'.join(args.test_dataset)}), ", end='')
    # if args.test_dataset2:
    #     print(f"test II: {len(test_loader2.dataset)} ({'/'.join(args.test_dataset2)}), ", end='')
    print(f"{input_type_name} - dimensions: {dimensions}")

    # Use reflection to load class for specified activation, optimizer data set
    if use_cuda and args.optimizer == 'Adam':
        Optimizer = locate('apex.optimizers.FusedAdam')
    else:
        Optimizer = locate('torch.optim.' + args.optimizer)
    Model = locate(module['module'] + '.' + args.cnn)
    if not Model:
        raise RuntimeError("Model " + args.cnn + " not found\n")
    if module['dim'] > 1 and module['min_input'] > dimensions[2]:
        model = Model(pretrained=False, num_classes=num_classes, num_channels=dimensions[0],
                      padding=(module['min_input'] - dimensions[2] + 1) // 2)
    else:
        model = Model(pretrained=False, num_classes=num_classes, num_channels=dimensions[0])
    if args.add_logsoftmax:
        model = nn.Sequential(model, nn.LogSoftmax(dim=1))
    if args.add_softmax:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    # def init_weights(m):
    #     """
    #     Explicit weight inititialization:
    #         * 0 for biases [nn.init.zeros_()]
    #         * He initialization for ReLU family [nn.init.kaiming_uniform_()]
    #         * Xavier for most others (tanh, identity, etc) [nn.init.xavier_uniform_()]
    #     The default for 'Linear' would otherwise be:
    #         self.weight.data.uniform_(-stdv, stdv)
    #     """
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1.0)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)

    # model.apply(init_weights)

    if args.tb_log:
        dummy_input = torch.autograd.Variable(torch.randn((1, ) + dimensions))
        writer = SummaryWriter(comment='')
        writer.add_text('config', str(args))
        writer.add_graph(model, (dummy_input, ))

    # Switch to target device after exporting
    model = model.to(device)
    if not args.no_parallel:
        # Distributed parallelism
        if ngpu > 1:
            print("Parallelizing for", ngpu, "GPUs")
            model = nn.DataParallel(model)
            # model = nn.parallel.DistributedDataParallel(model)
        # else:  # Run on CPU
        #     torch.distributed.init_process_group(world_size=4, init_method='...')
        #     nn.parallel.DistributedDataParallelCPU(module)
    # dummy_input = dummy_input.to(device)

    checkpoint_file = args.input_type
    checkpoint_file += '_' + str(dimensions[0])
    for dim in range(1, len(dimensions)):
        checkpoint_file += 'x' + str(dimensions[dim])
    checkpoint_file += '_' + args.loss_function
    if args.checkpoint != '':
        checkpoint_file += '-' + args.checkpoint
    checkpoint_file += '_' + args.cnn + '.pt'

    compression_scheduler = None
    optimizer_args = {}
    selected_optimizer = next(item for item in supported_optimizers
                              if item['name'] == args.optimizer)
    if selected_optimizer['momentum']:
        optimizer_args['momentum'] = args.momentum
    if selected_optimizer['centered']:
        optimizer_args['centered'] = True

    if (args.weight_decay_threshold is not None or args.weight_decay is not None) \
       and not selected_optimizer['weight_decay']:
        raise RuntimeError("Selected optimizer does not support weight decay, but "
                           "--weight-decay or --weight-decay-threshold specified")

    if args.resume:  # Load checkpoint
        if args.resume_best:
            fn = 'best_' + checkpoint_file
        else:
            fn = checkpoint_file
        print(f"Resuming from checkpoint {fn}...")
        assert os.path.isdir(args.checkpoint_directory), "Error: no checkpoint directory found!"
        checkpoint = torch.load(os.path.join(args.checkpoint_directory, fn),
                                map_location=device)

        best_accuracy = checkpoint['acc']
        best_loss = checkpoint['loss']
        eval_accuracy = checkpoint['last_acc']
        eval_loss = checkpoint['last_loss']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']
        if 'weight_decay_hysteresis' in checkpoint:
            adjust_weight_decay_hysteresis = checkpoint['weight_decay_hysteresis']

        if args.lr is None:
            optimizer_args['lr'] = checkpoint.get('lr', args.lr)
        else:
            optimizer_args['lr'] = args.lr
        start_epoch = checkpoint['epoch'] + 1

        if 'compression_sched' in checkpoint:
            compression_scheduler = distiller.CompressionScheduler(model, device=device)
            compression_scheduler.load_state_dict(checkpoint['compression_sched'])
            print(f"Loaded compression schedule from checkpoint "
                  f"(epoch {checkpoint['epoch']})")
        else:
            print("Warning: compression schedule data does not exist in checkpoint")

        if 'thinning_recipes' in checkpoint:
            if 'compression_sched' not in checkpoint:
                raise KeyError("Found thinning_recipes key, but missing mandatory key "
                               "'compression_sched'")
            # Cache the recipes in case we need them later
            model.thinning_recipes = checkpoint['thinning_recipes']
            distiller.execute_thinning_recipes_list(model,
                                                    compression_scheduler.zeros_mask_dict,
                                                    model.thinning_recipes)
            print("Loaded a thinning recipe from checkpoint")

        if 'quantizer_metadata' in checkpoint:
            qmd = checkpoint['quantizer_metadata']
            quantizer = qmd['type'](model, **qmd['params'])
            quantizer.prepare_model()
            print("Loaded quantizer metadata from checkpoint")

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint['net'])

        def set_grad(m):
            """
            Force the `requires_grad` flag on all weights and biases
            """
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                m.weight.requires_grad_()
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_()

        model.apply(set_grad)

        if 'optimizer' in checkpoint:
            optimizer = Optimizer(model.parameters(), **optimizer_args)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer from checkpoint")
            if args.lr is not None:  # Override learning rate
                optimizer.param_groups[0]['lr'] = args.lr
            if 'max_grad_norm' not in optimizer.param_groups[0]:
                optimizer.param_groups[0]['max_grad_norm'] = 0.0  # Apex FusedAdam resume from Adam
            if 'bias_correction' not in optimizer.param_groups[0]:
                optimizer.param_groups[0]['bias_correction'] = False

        if selected_optimizer['weight_decay']:
            if 'weight_decay' in checkpoint:
                weight_decay = optimizer_args['weight_decay'] = checkpoint['weight_decay']
                adjusting_weight_decay = checkpoint['adjusting_weight_decay']
            else:
                weight_decay = optimizer_args['weight_decay'] = DEFAULT_LEGACY_WEIGHT_DECAY
                adjusting_weight_decay = False

        del checkpoint  # Free memory

        print(f"Resuming from epoch {start_epoch}: validation accuracy: "
              f"{100. * eval_accuracy:.1f}%, ", end='')
        print(f"validation loss: {eval_loss:.2f}. "
              f"Best epoch {best_epoch + 1}: validation accuracy: {100. * best_accuracy:.1f}%, ",
              end='')
        print(f"validation loss: {best_loss:.2f}")
    else:
        optimizer_args['lr'] = args.lr if args.lr is not None else DEFAULT_LR
        if args.epochs == 0:
            raise NotImplementedError("--epochs 0 requires --resume")

    if args.trace:
        model.eval()
        dummy_input = torch.autograd.Variable(torch.randn((1, ) + dimensions)).to(device)

        print(distiller.model_summary(model, 'model'))

        print(distiller.model_performance_tbl_summary(model, dummy_input, batch_size=1))
        print(distiller.weights_sparsity_tbl_summary(model, param_dims=[1, 2, 4]))

        # print("\nNetwork input and output sizes")
        # print('-' * term_width)
        # trace_model(dummy_input, model)
        # print('')
        return

    # Override weight decay from command line
    if args.weight_decay is not None:
        weight_decay = optimizer_args['weight_decay'] = args.weight_decay

    if args.weight_decay_threshold and weight_decay == 0.:
        raise RuntimeError("--weight-decay-threshold specified, but weight decay is 0")

    if optimizer is None:
        optimizer = Optimizer(model.parameters(), **optimizer_args)

    # Adapt learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience)
    # See https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate

    if args.print_model:
        print("Model")
        print('-' * term_width)
        print(model)
        print('-' * term_width)
        print('')

    # Amp
    if use_cuda and not args.no_amp:
        # O0: FP32 training
        # O1: Conservative Mixed Precision
        # O2: Fast Mixed Precision
        # O3: FP16 training
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        use_amp = True
    else:
        use_amp = False

    def accuracy_classify(pred, target):
        """
        Definition of accurate inference when using classification
        """
        return (target == torch.argmax(pred, dim=1)).float()

    accuracy = accuracy_classify

    class_headers = []
    tabulate.MIN_PADDING = 0
    for i in range(num_classes):
        class_headers.append(labels[i])
    class_index = class_headers.copy()

    if args.compress and not compression_scheduler:
        # if args.resume:
        #     # Reset best-of state
        #     best_acc = 0.0
        #     best_top1 = None

        # Compression requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.config.file_config(model, optimizer, args.compress,
                                                             device)
        print('')

    Loss = locate('torch.nn.' + args.loss_function + 'Loss')
    if not Loss:
        raise RuntimeError("Loss function " + args.loss_function + " not found\n")
    # Loss function
    criterion = Loss(reduction='none').to(device)

    if args.summary_log:
        summary_log = open(args.summary_log, 'a')
        if start_epoch == 0:
            summary_log.write('Epoch,Type,LR,Weight Decay,Training Loss,Training Accuracy,'
                              'Verification Loss,Verification Accuracy')
            if args.test_dataset:
                summary_log.write(',Test Loss,Test Accuracy')
            if args.test_dataset2:
                summary_log.write(',Test2 Loss,Test2 Accuracy')
            summary_log.write('\n')

    def gpu_mem(use_cuda):
        """
        Return measure of GPU memory utilization.
        """
        return torch.cuda.max_memory_allocated(device) / 1024**3 if use_cuda else 0

    def train(epoch, _log, step):
        """
        Train the model
        """
        # Switch to train mode
        model.train()

        total_loss = 0.0
        running_loss = 0.0
        total_accuracy = 0.0

        log_time = start_time = time.time()
        processed = 0
        log_processed = 0

        total_samples = len(train_loader.dataset)
        steps_per_epoch = (total_samples + (train_loader.batch_size - 1)) \
            // train_loader.batch_size

        print('\nTraining:', flush=True)
        if not use_stdout:
            progress_status = progressbar.FormatCustomText(" Batch Loss:    N/A"
                                                           " Loss:    N/A"
                                                           " Accuracy:   N/A"
                                                           " GPU:   N/A",
                                                           dict(batch_loss=0.0,
                                                                avg_loss=0.0,
                                                                avg_accuracy=0.0,
                                                                gpu_memory=0.0),
                                                           new_style=True)
            running_progress_status = " Batch Loss: {batch_loss:.2f}" \
                                      " Loss: {avg_loss:.2f}" \
                                      " Accuracy: {avg_accuracy:.1f}%" \
                                      " GPU: {gpu_memory:2.1f}G"
            progress = progressbar.ProgressBar(max_value=total_samples,
                                               widgets=[progressbar.Percentage(), ' ',
                                                        progressbar.SimpleProgress(), ' ',
                                                        progressbar.Bar(fill='-'), ' ',
                                                        progressbar.Timer(), ' ',
                                                        progressbar.AdaptiveETA(),
                                                        progress_status]).start()

        for batch_idx, (data, target) in enumerate(train_loader):
            data_len = len(data)
            processed += data_len
            # img = data
            data, target = data.to(device), target.to(device)

            # Forward pass
            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch, batch_idx, steps_per_epoch,
                                                         optimizer)
            output = model(data)
            del data
            loss = criterion(output, target)
            total_accuracy += accuracy(output, target).sum()
            del target

            # Clamp loss?
            if args.loss_threshold:
                nn.functional.threshold(loss, args.loss_threshold, 0., inplace=True)
            if args.loss_function == 'KLDiv':
                loss = loss.sum() / data_len  # batchmean
            else:
                loss = loss.mean()
            total_loss += float(loss.item()) * data_len
            running_loss += float(loss.item()) * data_len

            if compression_scheduler:
                # Before the backward phase, add any regularization loss computed by the scheduler
                regularizer_loss = compression_scheduler.before_backward_pass(epoch, batch_idx,
                                                                              steps_per_epoch,
                                                                              loss, optimizer)
                running_loss += float(regularizer_loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            if not use_amp:
                loss.backward()  # do not set retain_graph
            else:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            optimizer.step()
            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch, batch_idx, steps_per_epoch,
                                                       optimizer)

            now = time.time()
            if now - log_time > args.log_interval and batch_idx < len(train_loader) - 1:
                log_time = now

                if not use_stdout:
                    progress_status.format = running_progress_status
                    progress_status.update_mapping(batch_loss=running_loss /
                                                   (processed - log_processed),
                                                   avg_loss=total_loss / processed,
                                                   avg_accuracy=100. * total_accuracy / processed,
                                                   gpu_memory=gpu_mem(use_cuda))
                    progress.update(batch_idx * data_len)
                else:
                    eta = datetime.timedelta(seconds=int((now - start_time)
                                                         * (total_samples - processed)
                                                         / processed))
                    print(f"{int(100. * processed / total_samples)}"
                          f"% {processed} of {total_samples} "
                          f"Batch Loss: {running_loss / (processed - log_processed):.2f} "
                          f"Loss: {total_loss / processed:.2f} "
                          f"Accuracy: {100. * total_accuracy / processed:.1f}% "
                          f"Time: {datetime.timedelta(seconds=int(now - start_time))} "
                          f"ETA: {eta} "
                          f"GPU: {gpu_mem(use_cuda):.1f}G", flush=True)

                # if log:
                #     # Log scalar values
                #     writer.add_scalar('train/loss', total_loss / processed, step)
                #     writer.add_scalar('train/batch_loss', running_loss /
                #                       (processed - log_processed), step)
                #     writer.add_scalar('train/accuracy', total_accuracy / processed, step)
                #     # Log values and gradients of the parameters (histogram summary)
                #     for name, param in model.named_parameters():
                #         name = name.replace('.', '/')
                #         writer.add_histogram(name, param, step, bins='doane')
                #         writer.add_histogram(name+'/grad', param.grad, step, bins='doane')
                #     # Log training images (image summary)
                #     writer.add_images('train/input', img, step)
                #     # Log embedding
                #     # writer.add_embedding(data, metadata=target, label_img=img.float(),
                #     #                      global_step=step, tag='train/data')

                # Reset running loss
                running_loss = 0.0
                log_processed = processed

            step = step + 1

        total_accuracy /= total_samples
        total_loss /= total_samples
        running_loss /= total_samples - log_processed

        if not use_stdout:
            progress_status.format = running_progress_status
            progress_status.update_mapping(batch_loss=running_loss,
                                           avg_loss=total_loss,
                                           avg_accuracy=100. * total_accuracy,
                                           gpu_memory=gpu_mem(use_cuda))
            progress.update(total_samples)
            progress.finish()
        else:
            now = time.time()
            print(f"100% {total_samples} of {total_samples} "
                  f"Batch Loss: {running_loss:.2f} "
                  f"Loss: {total_loss:.2f} "
                  f"Accuracy: {100. * total_accuracy:.1f}% "
                  f"Time: {datetime.timedelta(seconds=int(now - start_time))} "
                  f"ETA: 0:00:00 GPU: {gpu_mem(use_cuda):.1f}G")
        sys.stdout.flush()

        if args.summary_log:
            summary_log.write(f"{epoch+1},{input_type_name},"
                              f"{optimizer.param_groups[0]['lr']},{weight_decay},"
                              f"{total_loss:.2f},{100. * total_accuracy:.2f}%")

        return step, total_loss

    def evaluate(_epoch, loader, log, step, header='Evaluation'):
        """
        Evaluate the model
        """
        # Switch to evaluate mode
        model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_accuracy = 0.0
            processed = 0
            conf_matrix = np.zeros(shape=(len(class_index), len(class_headers)), dtype=int)

            log_time = start_time = time.time()
            total_samples = len(loader.dataset)

            print('\n' + header + ':', flush=True)
            if not use_stdout:
                progress_status = progressbar.FormatCustomText(" Loss:      N/A"
                                                               " Accuracy:    N/A"
                                                               " GPU:   N/A",
                                                               dict(avg_loss=0.0,
                                                                    avg_accuracy=0.0,
                                                                    gpu_memory=0.0),
                                                               new_style=True)
                running_progress_status = " Loss: {avg_loss:.2f}" \
                                          " Accuracy: {avg_accuracy:.1f}%" \
                                          " GPU: {gpu_memory:2.1f}G"
                progress = progressbar.ProgressBar(max_value=total_samples,
                                                   widgets=[progressbar.Percentage(), ' ',
                                                            progressbar.SimpleProgress(), ' ',
                                                            progressbar.Bar(fill='-'), ' ',
                                                            progressbar.Timer(), ' ',
                                                            progressbar.AdaptiveETA(),
                                                            progress_status]).start()

            for batch_idx, (data, target) in enumerate(loader):
                data_len = len(data)
                processed += data_len
                data, target = data.to(device), target.to(device)
                output = model(data)
                del data
                if args.loss_function == 'KLDiv':
                    loss = criterion(output, target).sum() / data_len  # batchmean
                else:
                    loss = criterion(output, target).mean()
                total_accuracy += accuracy(output, target).sum()

                pred = output.argmax(1, keepdim=True)  # Get index of the max probability
                # Update confusion matrix
                for a, p in zip(target.view_as(pred), pred):
                    conf_matrix[a, p] += 1
                del target
                total_loss += float(loss.item()) * data_len

                now = time.time()
                if now - log_time > args.log_interval and batch_idx < len(loader) - 1:
                    log_time = now

                    if not use_stdout:
                        progress_status.format = running_progress_status
                        progress_status.update_mapping(avg_loss=total_loss / processed,
                                                       avg_accuracy=100. * total_accuracy /
                                                       processed,
                                                       gpu_memory=gpu_mem(use_cuda))
                        progress.update(processed)
                    else:
                        eta = datetime.timedelta(seconds=int((now - start_time)
                                                             * (total_samples - processed)
                                                             / processed))
                        print(f"{int(100. * processed / total_samples)}% "
                              f"{processed} of {total_samples} "
                              f"Loss: {total_loss / processed:.2f} "
                              f"Accuracy: {100. * total_accuracy / processed:.1f}% "
                              f"Time: {datetime.timedelta(seconds=int(now - start_time))} "
                              f"ETA: {eta} "
                              f"GPU: {gpu_mem(use_cuda):.1f}G", flush=True)

            total_loss /= total_samples
            total_accuracy /= total_samples

            if not use_stdout:
                progress_status.format = running_progress_status
                progress_status.update_mapping(avg_loss=total_loss,
                                               avg_accuracy=100. * total_accuracy,
                                               gpu_memory=gpu_mem(use_cuda))
                progress.update(total_samples)
                progress.finish()
            else:
                now = time.time()
                print(f"100% {total_samples} of {total_samples} "
                      f"Loss: {total_loss:.2f} "
                      f"Accuracy: {100. * total_accuracy:.1f}% "
                      f"Time: {datetime.timedelta(seconds=int(now - start_time))} "
                      f"ETA: 0:00:00 GPU: {gpu_mem(use_cuda):.1f}G")
            sys.stdout.flush()

            print("\nConfusion matrix - " + header)
            print(tabulate.tabulate(conf_matrix, headers=class_headers, tablefmt='fancy_grid',
                                    showindex=class_index), flush=True)

            # print(f'{input_type_name}, Epoch {epoch+1}, Average loss: '
            #       f'{total_loss:.3f}, Accuracy: {100. * total_accuracy:.2f}%'
            #       f', ', flush=True)

            if log:
                writer.add_scalar('eval/loss', total_loss, step)
                writer.add_scalar('eval/accuracy', total_accuracy, step)

            if args.summary_log:
                summary_log.write(f",{total_loss:.3f},{100. * total_accuracy:.2f}%")

            return total_loss, total_accuracy

    if args.epochs == 0:
        _, _, _ = evaluate(start_epoch, eval_loader, args.tb_log, 0,
                           header='Evaluation (' + args.cv_dataset + ')')

    def save_state(epoch, prefix=''):
        fn = prefix + checkpoint_file
        print(f"Saving checkpoint, epoch {epoch+1} to {fn}...", flush=True)
        if isinstance(model, nn.DataParallel):
            mstate = model.module.state_dict()
        else:
            mstate = model.state_dict()
        state = {
            'net': mstate,
            'acc': best_accuracy,
            'last_acc': eval_accuracy,
            'loss': best_loss,
            'last_loss': eval_loss,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': weight_decay,
            'weight_decay_hysteresis': adjust_weight_decay_hysteresis,
            'adjusting_weight_decay': adjusting_weight_decay
        }
        if not os.path.isdir(args.checkpoint_directory):
            os.mkdir(args.checkpoint_directory)

        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if compression_scheduler is not None:
            state['compression_sched'] = compression_scheduler.state_dict()
        if hasattr(model, 'thinning_recipes'):
            state['thinning_recipes'] = model.thinning_recipes
        if hasattr(model, 'quantizer_metadata'):
            state['quantizer_metadata'] = model.quantizer_metadata

        torch.save(state, os.path.join(args.checkpoint_directory, fn))
        return epoch

    # Step used for logging
    step = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        if input_type_name != '2d':
            print(f"{input_type_name}, ", end='')
        print(f"Epoch {epoch+1}: "
              f"Learning rate {optimizer.param_groups[0]['lr']}, weight decay {weight_decay}",
              flush=True)
        if args.tb_log:
            # Plot learning rate (same for all layers, so use index 0)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)

        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        # Train
        step, train_loss = train(epoch, args.tb_log, step)

        if adjust_weight_decay_hysteresis > 0:
            adjust_weight_decay_hysteresis -= 1
        if args.weight_decay_threshold:
            if train_loss <= args.weight_decay_threshold and weight_decay < args.weight_decay_max:
                if adjust_weight_decay_hysteresis == 0:
                    adjusting_weight_decay = True
                    adjust_weight_decay_hysteresis = args.weight_decay_hysteresis + 1
                    weight_decay *= args.weight_decay_momentum
                    optimizer.param_groups[0]['weight_decay'] = weight_decay
                    print(f"Increasing weight decay to {weight_decay}", end='')

            elif adjusting_weight_decay and train_loss > args.weight_decay_threshold * 1.05 \
               and weight_decay > args.weight_decay_min:  # noqa: E127
                if adjust_weight_decay_hysteresis == 0:
                    adjust_weight_decay_hysteresis = args.weight_decay_hysteresis + 1
                    weight_decay /= args.weight_decay_momentum
                    optimizer.param_groups[0]['weight_decay'] = weight_decay
                    print(f"Decreasing weight decay to {weight_decay}", end='')

        print('')

        # Evaluate
        eval_loss, eval_accuracy = evaluate(epoch, eval_loader, args.tb_log, step,
                                            header='Validation ('
                                            + args.cv_dataset + ')')
        save_condition = eval_accuracy > best_accuracy

        # Also run through test set(s) (uncommon)
        # if args.test_dataset:
        #     evaluate(epoch, test_loader, False, step,
        #              header='Test (' + '/'.join(args.test_dataset) + ')')
        # if args.test_dataset2:
        #     evaluate(epoch, test_loader2, False, step,
        #              header='Test (' + '/'.join(args.test_dataset2) + ')')

        if args.summary_log:
            summary_log.write('\n')
            summary_log.flush()

        # Save checkpoint?
        if args.checkpoints and args.epochs > 0 and save_condition:
            best_loss = eval_loss
            best_accuracy = eval_accuracy
            best_epoch = epoch
            save_state(epoch, 'best_')
        if args.checkpoint_save_frequency is not None and \
           epoch % args.checkpoint_save_frequency == 0:
            last_saved_epoch = save_state(epoch)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        print('')
        if not args.lr_min or optimizer.param_groups[0]['lr'] > args.lr_min:
            scheduler.step(eval_loss)  # Adjust learning rate
            if args.lr_limit and optimizer.param_groups[0]['lr'] <= args.lr_limit:
                print(f"Learning rate {optimizer.param_groups[0]['lr']} at limit, terminating...")
                break

    if args.checkpoint_on_quit and epoch != last_saved_epoch:
        save_state(epoch)
        print('')

    if args.tb_log:
        writer.close()
    if args.summary_log:
        summary_log.close()

    # --------------------------------------------------------------------------------------------


def signal_handler(_signal, _frame):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    if torch.__version__ < '1.0.1':
        raise NotImplementedError(f"ERROR: This code requires PyTorch version 1.0.1 or higher. "
                                  f"You have version {torch.__version__}")
    main()
