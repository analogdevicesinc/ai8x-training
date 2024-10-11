#!/usr/bin/env python3
#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2019-2024 Maxim Integrated Products, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pyright: reportMissingModuleSource=false, reportGeneralTypeIssues=false
# pyright: reportOptionalSubscript=false
"""This is the example training application for MAX7800x.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
"""

import copy
import fnmatch
import logging
import operator
import os
import re

# pylint: disable=wrong-import-position
if os.name == 'posix':
    import resource  # pylint: disable=import-error
# pylint: enable=wrong-import-position

import shutil
import sys
import time
import traceback
from collections import OrderedDict
from pydoc import locate

import numpy as np

import matplotlib

# pylint: disable=wrong-import-position
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Set this before importing PyTorch
# pylint: enable=wrong-import-position

# TensorFlow 2.x compatibility
try:
    import tensorboard  # pylint: disable=import-error
    import tensorflow  # pylint: disable=import-error
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
except (ModuleNotFoundError, AttributeError):
    pass

import torch
import torch.distributed
import torch.optim
import torch.utils.data
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel

# pylint: disable=wrong-import-order
import distiller
import torchnet.meter as tnt
from distiller import apputils, model_summaries  # type: ignore[attr-defined]
from distiller.data_loggers import PythonLogger, TensorBoardLogger
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import testers
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

import ai8x
import ai8x_nas
import datasets
import nnplot
import parse_qat_yaml
import parsecmd
import sample
import yamlwriter
from losses.dummyloss import DummyLoss
from losses.multiboxloss import MultiBoxLoss
from nas import parse_nas_yaml
from utils import kd_relationbased, model_wrapper, object_detection_utils, parse_obj_detection_yaml

matplotlib.use("pgf")

# Logger handle
msglogger = None

# Globals
weight_min = None
weight_max = None
weight_count = None
weight_sum = None
weight_stddev = None
weight_mean = None


def main():
    """main"""
    script_dir = os.path.dirname(__file__)
    global msglogger  # pylint: disable=global-statement

    supported_models = []
    supported_sources = []
    model_names = []
    dataset_names = []

    local_rank = None
    local_world_size = 1
    try:
        local_rank = os.environ['LOCAL_RANK']
        local_world_size = os.environ['LOCAL_WORLD_SIZE']
    except KeyError:
        pass
    finally:
        local_rank = int(local_rank) if local_rank is not None else -1
        if local_world_size is not None:
            local_world_size = int(local_world_size)

    # Dynamically load models
    for _, _, files in sorted(os.walk('models')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                fn = 'models.' + name[:-3]
                m = locate(fn)
                try:
                    for i in m.models:
                        i['module'] = fn
                    supported_models += m.models
                    model_names += [item['name'] for item in m.models]
                except AttributeError:
                    # Skip files that don't have 'models' or 'models.name'
                    pass

    # Dynamically load datasets
    for _, _, files in sorted(os.walk('datasets')):
        for name in sorted(files):
            if fnmatch.fnmatch(name, '*.py'):
                ds = locate('datasets.' + name[:-3])
                try:
                    supported_sources += ds.datasets
                    dataset_names += [item['name'] for item in ds.datasets]
                except AttributeError:
                    # Skip files that don't have 'datasets' or 'datasets.name'
                    pass

    # Parse arguments
    args = parsecmd.get_parser(model_names, dataset_names).parse_args()
    args.local_world_size = local_world_size

    if local_rank <= 0:  # not DistributedDataParallel or rank 0
        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name,
                                             args.output_dir)
    else:
        msglogger = logging.getLogger()
        msglogger.log_filename = 'none'
        msglogger.setLevel(logging.CRITICAL)
        pattern = re.compile(r'.*Profiler function .* will be ignored')
        logging.getLogger('torch._dynamo.variables.torch').addFilter(
            lambda record: not pattern.search(record.getMessage())
        )

    # Redirect 'print'
    class StdStreamLogger():
        """Stream object to redirect sys.stdout to Python logger"""
        def __init__(self, logger, level):
            self._logger = logger
            self._level = level
            self._buf = ''

        def write(self, msg):
            """write()"""
            self._buf = self._buf + msg
            while '\n' in self._buf:
                pos = self._buf.find('\n')
                self._logger.log(self._level, self._buf[:pos])
                self._buf = self._buf[pos + 1:]

        def flush(self):
            """flush()"""
            if self._buf != '':
                self._logger.log(self._level, self._buf)
                self._buf = ''

    sys.stdout = StdStreamLogger(msglogger, logging.INFO)

    if os.name == 'posix':
        # Check file descriptor limits
        nfiles = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if nfiles < 4096:
            msglogger.warning('The open file limit is %d. '
                              'Please raise the limit (see documentation).', nfiles)

    # Set hardware device
    ai8x.set_device(args.device, args.act_mode_8bit, args.avg_pool_rounding)

    if args.epochs is None:
        args.epochs = 90

    if not os.path.exists(args.output_dir) and local_rank <= 0:  # not DDP or rank 0
        os.makedirs(args.output_dir)

    if args.optimizer is None:
        args.optimizer = 'SGD'
        if not args.evaluate:
            msglogger.warning('--optimizer not set, selecting %s.', args.optimizer)

    if args.lr is None:
        args.lr = 0.1
        if not args.evaluate:
            msglogger.warning('Initial learning rate (--lr) not set, selecting %f.', args.lr)

    if args.generate_sample is not None and not args.act_mode_8bit:
        msglogger.warning('Cannot save sample in training mode, ignoring --save-sample option. '
                          'Use with --evaluate instead.')

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    if local_rank <= 0:  # not DistributedDataParallel or rank 0
        apputils.log_execution_env_state(args.compress, msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    ending_epoch = args.epochs
    perf_scores_history = []

    if args.evaluate:
        args.deterministic = True
    if args.deterministic:
        # torch.set_deterministic(True)
        distiller.set_deterministic(args.seed)  # For experiment reproducibility
        if args.seed is not None:
            distiller.set_seed(args.seed)
    else:
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here:
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True

    if args.cpu or (not torch.cuda.is_available() and not torch.backends.mps.is_available()):
        if not args.cpu:
            # Print warning if no hardware acceleration
            msglogger.warning('No CUDA, ROCm, or MPS hardware acceleration, training will be slow')
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    elif torch.cuda.is_available():
        args.device = 'cuda'
        if local_rank >= 0:  # DistributedDataParallel
            torch.cuda.set_device(local_rank)
            args.gpus = local_rank
        elif args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError as exc:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated '
                                 'list of integers only') from exc
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError(f'ERROR: GPU device ID {dev_id} requested, but only '
                                     f'{available_gpus} devices available')
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])
    elif os.uname().release < '22.3.0':
        msglogger.warning('mps disabled, update macOS to Ventura 13.4 or later for mps support')
        args.device = 'cpu'
    else:
        args.device = 'mps'

    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    args.labels = selected_source['output']
    args.num_classes = len(args.labels)

    # Add background class explicitly for the object detection models
    if args.obj_detection:
        args.num_classes += 1

    if args.num_classes == 1 \
       or ('regression' in selected_source and selected_source['regression']):
        args.regression = True
    dimensions = selected_source['input']
    if len(dimensions) == 2:
        dimensions += (1, )
    args.dimensions = dimensions

    args.datasets_fn = selected_source['loader']
    args.collate_fn = selected_source.get('collate')  # .get returns None if key does not exist

    args.visualize_fn = selected_source['visualize'] \
        if 'visualize' in selected_source else datasets.visualize_data

    if (args.regression or args.obj_detection) and args.display_confusion:
        raise ValueError('ERROR: Argument --confusion cannot be used with regression '
                         'or object detection')
    if (args.regression or args.obj_detection) and args.display_prcurves:
        raise ValueError('ERROR: Argument --pr-curves cannot be used with regression '
                         'or object detection')
    if (args.regression or args.obj_detection) and args.display_embedding:
        raise ValueError('ERROR: Argument --embedding cannot be used with regression '
                         'or object detection')

    model = create_model(supported_models, dimensions, args)

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    pylogger = PythonLogger(msglogger, log_1d=True)
    all_loggers = [pylogger]
    if args.tblog and local_rank <= 0:  # not DistributedDataParallel or rank 0
        tflogger = TensorBoardLogger(msglogger.logdir, log_1d=True, comment='_'+args.dataset)

        tflogger.tblogger.writer.add_text('Command line', str(args))

        if dimensions[2] > 1:
            dummy_input = torch.randn((1, ) + dimensions)
        else:  # 1D input
            dummy_input = torch.randn((1, ) + dimensions[:-1])
        tflogger.tblogger.writer.add_graph(model.to('cpu'), (dummy_input, ), False)

        all_loggers.append(tflogger)
        all_tbloggers = [tflogger]
    else:
        tflogger = None
        all_tbloggers = []

    # Get policy for quantization aware training
    qat_policy = parse_qat_yaml.parse(args.qat_policy) \
        if args.qat_policy.lower() != "none" else None

    # Get policy for once for all training policy
    nas_policy = parse_nas_yaml.parse(args.nas_policy) \
        if args.nas and args.nas_policy.lower() != '' else None

    # Get object detection params
    obj_detection_params = parse_obj_detection_yaml.parse(args.obj_detection_params) \
        if args.obj_detection_params else None

    # We can optionally resume from a checkpoint
    optimizer = None
    loss_optimizer = None
    if args.resumed_checkpoint_path:
        if qat_policy is not None:
            checkpoint = torch.load(args.resumed_checkpoint_path,
                                    map_location=lambda storage, loc: storage)
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                ai8x.fuse_bn_layers(model)
                if args.name:
                    args.name = f'{args.name}_qat'
                else:
                    args.name = 'qat'
        try:
            model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
                model, args.resumed_checkpoint_path, model_device=args.device)
        except ValueError as exc:
            raise ValueError('\n ERROR: Unable to resume from the checkpoint. '
                             'The reason might be the size mismatch between checkpoint and'
                             ' optimizer. Instead of "--resume-from", "--exp-load-weights-from" '
                             'argument can be used to load the lean model. ') from exc
    elif args.load_model_path:
        init_qat = False
        update_old_model_params(args.load_model_path, model)
        if qat_policy is not None:
            checkpoint = torch.load(args.load_model_path,
                                    map_location=lambda storage, loc: storage)
            if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
                init_qat = True
                ai8x.fuse_bn_layers(model)
                if args.name:
                    args.name = f'{args.name}_qat'
                else:
                    args.name = 'qat'
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)

        # If model is in QAT mode, guarantee that the model is initialized for QATv2
        if init_qat:
            ai8x.initiate_qat(model, qat_policy)

    ai8x.update_model(model)

    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and '
                           'resetting epoch count to 0')

    # Define loss function (criterion)
    if args.obj_detection:
        criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy,
                                 alpha=obj_detection_params['multi_box_loss']['alpha'],
                                 neg_pos_ratio=obj_detection_params['multi_box_loss']
                                 ['neg_pos_ratio'], device=args.device).to(args.device)
    elif args.dr:
        criterion = pml_losses.SubCenterArcFaceLoss(num_classes=args.num_classes,
                                                    embedding_size=args.dr,
                                                    margin=args.scaf_margin,
                                                    scale=args.scaf_scale)
        if args.resumed_checkpoint_path:
            checkpoint = torch.load(args.resumed_checkpoint_path,
                                    map_location=lambda storage, loc: storage)
            criterion.W = checkpoint['extras']['loss_weights']
        criterion = criterion.to(args.device)

        loss_optimizer = torch.optim.Adam(criterion.parameters(), lr=args.scaf_lr)
        if args.resumed_checkpoint_path:
            loss_optimizer.load_state_dict(checkpoint['extras']['loss_optimizer_state_dict'])

        distance_fn = CosineSimilarity()
        custom_knn = CustomKNN(distance_fn, batch_size=args.batch_size)
        accuracy_calculator = AccuracyCalculator(knn_func=custom_knn,
                                                 include=("precision_at_1",), k=1)
    elif args.regression:
        criterion = nn.MSELoss().to(args.device)
    elif 'weight' in selected_source:
        criterion = nn.CrossEntropyLoss(
            torch.tensor(selected_source['weight'], dtype=torch.float)
        ).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss().to(args.device)

    # Override criterion with dummy loss when student weight is 0
    if args.kd_student_wt == 0:
        criterion = DummyLoss(device=args.device).to(args.device)
        msglogger.info("WARNING: kd_student_wt == 0, Overwriting criterion with a dummy loss")

    if optimizer is None:
        optimizer = create_optimizer(model, args)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary,
                               filename=args.summary_filename)
    if args.yaml_template is not None:
        return yamlwriter.create(
            model,
            args.dataset,
            args.cnn,
            filename=args.yaml_template,
            qat_policy=qat_policy,
        )

    if local_rank >= 0:  # DistributedDataParallel
        torch.distributed.init_process_group(backend='nccl' if args.device == 'cuda' else 'gloo')
        model = DistributedDataParallel(
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(model),
            device_ids=[local_rank] if args.device == 'cuda' else None,
            output_device=local_rank if args.device == 'cuda' else None,
        )

    if local_rank >= 0:  # DistributedDataParallel
        # Auto-divide batch size
        args.batch_size //= local_world_size
        # Create semaphore
        tensor = torch.zeros(1).to(args.device)
    if local_rank > 0:  # DistributedDataParallel and rank > 0
        # Wait for rank 0 to take advantage of disk caching and downloading
        torch.distributed.broadcast(tensor, src=0)

    # Load the datasets
    train_loader, val_loader, test_loader, _, train_sampler = apputils.get_data_loaders(
        args.datasets_fn, (os.path.expanduser(args.data), args), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        1., 1., 1.,  # effective sizes 100%
        test_only=args.evaluate, collate_fn=args.collate_fn, cpu=args.device == 'cpu',
        distributed=local_rank >= 0, rank=local_rank, world_size=local_world_size)
    assert args.evaluate or train_loader is not None and val_loader is not None, \
        "No training and/or validation data in train mode"
    assert not args.evaluate or test_loader is not None, "No test data in eval mode"
    assert local_rank < 0 or train_sampler is not None

    if local_rank == 0:  # DistributedDataParallel rank 0
        # Notify the other ranks but don't wait
        torch.distributed.broadcast(tensor, src=0, async_op=True)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress,
                                                      compression_scheduler,
                                                      (start_epoch-1)
                                                      if args.resumed_checkpoint_path
                                                      else None, loss_optimizer)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
    model.to(args.device)

    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(supported_models, dimensions, args, mode='kd_teacher')
        if args.kd_resume:
            teacher = apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt,
                                                args.kd_teacher_wt)
        if args.kd_relationbased:
            args.kd_policy = kd_relationbased.RelationBasedKDPolicy(model, teacher,
                                                                    dlw, args.act_mode_8bit)
        else:
            args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher,
                                                                   args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch,
                                         ending_epoch=args.epochs, frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join([f'{val:.2f}' for val in dlw]))
        msglogger.info('\tStarting from Epoch: %d', args.kd_start_epoch)

    if start_epoch >= ending_epoch:
        msglogger.error('epoch count is too low, starting epoch is %d but total epochs set '
                        'to %d', start_epoch, ending_epoch)
        raise ValueError('Epochs parameter is too low. Nothing to do.')

    if args.nas:
        assert isinstance(model, ai8x_nas.OnceForAllModel), 'Model should implement ' \
            'OnceForAllModel interface for NAS training!'
        if nas_policy:
            args.nas_stage_transition_list = create_nas_training_stage_list(model, nas_policy)
            args.nas_kd_params = nas_policy['kd_params'] \
                if nas_policy and 'kd_params' in nas_policy else None
            if args.nas_kd_resume_from == '':
                args.nas_kd_policy = None
            else:
                if args.nas_kd_params['teacher_model'] == 'full_model':
                    kd_end_epoch = args.epochs
                else:
                    kd_end_epoch = get_next_stage_start_epoch(start_epoch,
                                                              args.nas_stage_transition_list,
                                                              args.epochs)
                create_nas_kd_policy(model, compression_scheduler, start_epoch, kd_end_epoch, args)

    if args.compiler_mode.lower() != 'none':
        if local_rank >= 0 or args.device == 'cuda' \
           and (torch.cuda.device_count() == 1 or args.gpus is not None and len(args.gpus) <= 1):
            model = torch.compile(model, mode=args.compiler_mode,
                                  backend=args.compiler_backend)
            msglogger.info(
                'torch.compile() successful, mode=%s, cache limit=%d',
                args.compiler_mode,
                torch._dynamo.config.cache_size_limit,  # pylint: disable=protected-access
            )
        else:
            msglogger.info('torch.compile() not available, using "eager" mode')
            if args.device == 'cuda' and torch.cuda.device_count() > 1:
                msglogger.info('Use distributed training to enable torch.compile() '
                               'with multiple GPUs')

    if args.evaluate:
        msglogger.info('Dataset sizes:\n\ttest=%d', len(test_loader.sampler))
        return test(test_loader, model, criterion, pylogger, args=args)

    assert train_loader and val_loader
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    vloss = 10**6
    for epoch in range(start_epoch, ending_epoch):
        if local_rank >= 0:  # DistributedDataParallel
            train_sampler.set_epoch(epoch)

        if qat_policy is not None and epoch > 0 and epoch == qat_policy['start_epoch']:
            msglogger.info('Initiating quantization aware training (QAT)...')

            model, dynamo, ddp = model_wrapper.unwrap(model)

            # Fuse the BN parameters into conv layers before Quantization Aware Training (QAT)
            ai8x.fuse_bn_layers(model)
            ai8x.init_hist(model)

            msglogger.info('Collecting statistics for quantization aware training (QAT)...')
            stat_collect(train_loader, model, args)

            ai8x.init_threshold(model, qat_policy["outlier_removal_z_score"])
            ai8x.release_hist(model)

            ai8x.apply_scales(model)

            # Update the optimizer to reflect fused batchnorm layers
            optimizer = ai8x.update_optimizer(model, optimizer)

            # Update the compression scheduler to reflect the updated optimizer
            for ep, _ in enumerate(compression_scheduler.policies):
                for pol in compression_scheduler.policies[ep]:
                    for attr_key in dir(pol):
                        attr = getattr(pol, attr_key)
                        if hasattr(attr, 'optimizer'):
                            attr.optimizer = optimizer

            # Switch model from unquantized to quantized for QAT
            ai8x.initiate_qat(model, qat_policy)

            # Model is re-transferred to GPU in case parameters were added
            model.to(args.device)

            if ddp:
                model = DistributedDataParallel(
                    model,
                    device_ids=[local_rank] if args.device == 'cuda' else None,
                    output_device=local_rank if args.device == 'cuda' else None,
                )

            if dynamo:
                torch._dynamo.reset()  # pylint: disable=protected-access
                model = torch.compile(model, mode=args.compiler_mode,
                                      backend=args.compiler_backend)
                msglogger.info(
                    'torch.compile() successful, mode=%s, cache limit=%d',
                    args.compiler_mode,
                    torch._dynamo.config.cache_size_limit,  # pylint: disable=protected-access
                )

            # Empty the performance scores list for QAT operation
            perf_scores_history = []
            if args.name:
                args.name = f'{args.name}_qat'
            else:
                args.name = 'qat'

        # This is the main training loop.
        msglogger.info('\n')
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch, metrics=vloss)

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
              loggers=all_loggers, args=args, loss_optimizer=loss_optimizer)

        # evaluate on validation set
        run_validation = not args.nas or (args.nas and (epoch < nas_policy['start_epoch']))
        run_nas_validation = args.nas and (epoch >= nas_policy['start_epoch']) and \
            ((epoch+1) % nas_policy['validation_freq'] == 0)

        if run_validation or run_nas_validation:
            checkpoint_name = args.name

            # run pre validation steps if NAS is running
            if run_nas_validation:
                update_bn_stats(train_loader, model, args)
                stage, level = get_nas_training_stage(epoch, args.nas_stage_transition_list)
                if args.name:
                    checkpoint_name = f'{args.name}_nas_stg{stage}_lev{level}'
                else:
                    checkpoint_name = f'nas_stg{stage}_lev{level}'

            if not args.dr:
                top1, top5, vloss, mAP = validate(val_loader, model, criterion, [pylogger],
                                                  args, epoch, tflogger)
            else:
                top1, top5, vloss, mAP = scaf_test(val_loader, model, accuracy_calculator)

            if args.obj_detection:
                stats = ('Performance/Validation/', OrderedDict([('Loss', vloss),
                                                                 ('mAP', mAP)]))
            elif args.regression:
                stats = ('Performance/Validation/', OrderedDict([('Loss', vloss),
                                                                 ('MSE', top1)]))
            else:
                stats = ('Performance/Validation/', OrderedDict([('Loss', vloss),
                                                                 ('Top1', top1)]))
                if args.num_classes > 5 and not args.dr:
                    stats[1]['Top5'] = top5

            distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1,
                                            log_freq=1, loggers=all_tbloggers)

            # Update the list of top scores achieved so far
            update_training_scores_history(perf_scores_history, model, top1, top5, mAP, vloss,
                                           epoch, args)

            # Save the checkpoint
            if run_validation:
                is_best = epoch == perf_scores_history[0].epoch
                checkpoint_extras = {'current_top1': top1,
                                     'best_top1': perf_scores_history[0].top1,
                                     'current_mAP': mAP,
                                     'best_mAP': perf_scores_history[0].mAP,
                                     'best_epoch': perf_scores_history[0].epoch}
            else:
                is_best = False
                checkpoint_extras = {'current_top1': top1,
                                     'current_mAP': mAP}
            if args.dr:
                checkpoint_extras['loss_weights'] = criterion.W
                checkpoint_extras['loss_optimizer_state_dict'] = loss_optimizer.state_dict()

            if local_rank <= 0:  # not DistributedDataParallel or rank 0
                m, _, _ = model_wrapper.unwrap(model)
                apputils.save_checkpoint(epoch, args.cnn, m, optimizer=optimizer,
                                         scheduler=compression_scheduler, extras=checkpoint_extras,
                                         is_best=is_best, name=checkpoint_name,
                                         dir=msglogger.logdir)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

    # Finally run results on the test set
    if not args.dr:
        test(test_loader, model, criterion, [pylogger], args=args, mode="ckpt")
        test(test_loader, model, criterion, [pylogger], args=args, mode="best",
             ckpt_name=checkpoint_name)

    if args.copy_output_folder and local_rank <= 0:
        msglogger.info('Copying output folder to: %s', args.copy_output_folder)
        shutil.copytree(msglogger.logdir, args.copy_output_folder, dirs_exist_ok=True)

    return None


OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def create_model(supported_models, dimensions, args, mode='default'):
    """Create the model"""
    if mode == 'kd_teacher':
        module = next(item for item in supported_models if item['name'] == args.kd_teacher)
    else:  # including 'default'
        module = next(item for item in supported_models if item['name'] == args.cnn)

    # Override distiller's input shape detection. This is not a very clean way to do it since
    # we're replacing a protected member.
    distiller.utils._validate_input_shape = (  # pylint: disable=protected-access
        lambda _a, _b: (1, ) + dimensions[:module['dim'] + 1]
    )
    if mode == 'kd_teacher':
        Model = locate(module['module'] + '.' + args.kd_teacher)
        if not Model:
            raise RuntimeError("Model " + args.kd_teacher + " not found\n")

    else:  # including 'default'
        Model = locate(module['module'] + '.' + args.cnn)
        if not Model:
            raise RuntimeError("Model " + args.cnn + " not found\n")

    if args.dr and ('dr' not in module or not module['dr']):
        raise ValueError("Dimensionality reduction is not supported for this model")

    # Set model parameters
    if args.act_mode_8bit:
        weight_bits = 8
        bias_bits = 8
        quantize_activation = True
    else:
        weight_bits = None
        bias_bits = None
        quantize_activation = False

    model_args = {}
    model_args["pretrained"] = False
    model_args["num_classes"] = args.num_classes
    model_args["num_channels"] = dimensions[0]
    model_args["dimensions"] = (dimensions[1], dimensions[2])
    model_args["bias"] = args.use_bias
    model_args["weight_bits"] = weight_bits
    model_args["bias_bits"] = bias_bits
    model_args["quantize_activation"] = quantize_activation

    if args.dr:
        model_args["dimensionality"] = args.dr

    if args.backbone_checkpoint:
        model_args["backbone_checkpoint"] = args.backbone_checkpoint

    if args.obj_detection:
        model_args["device"] = args.device

    if module['dim'] > 1 and module['min_input'] > dimensions[2]:
        model_args["padding"] = (module['min_input'] - dimensions[2] + 1) // 2

    model = Model(**model_args).to(args.device)

    return model


def create_optimizer(model, args):
    """Create the optimizer"""
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        assert msglogger is not None
        msglogger.info('Unknown optimizer type: %s. SGD is set as optimizer!!!', args.optimizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    return optimizer


def create_nas_kd_policy(model, compression_scheduler, epoch, next_state_start_epoch, args):
    """Create knowledge distillation policy for nas"""
    teacher = copy.deepcopy(model)
    dlw = distiller.DistillationLossWeights(args.nas_kd_params['distill_loss'],
                                            args.nas_kd_params['student_loss'], 0)
    args.nas_kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher,
                                                               args.nas_kd_params['temperature'],
                                                               dlw)
    compression_scheduler.add_policy(args.nas_kd_policy, starting_epoch=epoch,
                                     ending_epoch=next_state_start_epoch, frequency=1)

    assert msglogger is not None
    msglogger.info('\nStudent-Teacher knowledge distillation enabled for NAS:')
    msglogger.info('\tStart Epoch: %d, End Epoch: %d', epoch, next_state_start_epoch)
    msglogger.info('\tTemperature: %s', args.nas_kd_params['temperature'])
    msglogger.info("\tLoss Weights (distillation | student | teacher): %s",
                   ' | '.join([f'{val:.2f}' for val in dlw]))


@torch.no_grad()
def stat_collect(train_loader, model, args):
    """Collect statistics for quantization aware training"""
    model.eval()
    for inputs, _ in tqdm(train_loader):
        inputs = inputs.to(args.device)
        model(inputs)


def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args, loss_optimizer=None):
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    if not args.regression:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(args.num_classes, 5)))
    else:
        classerr = tnt.MSEMeter()
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = (total_samples + batch_size - 1) // batch_size
    assert msglogger is not None
    msglogger.info('Training epoch: %d samples (%d per mini-batch, world size: %d)',
                   total_samples, batch_size, args.local_world_size)

    if args.nas:
        if args.nas_stage_transition_list is not None:
            stage, level = get_nas_training_stage(epoch, args.nas_stage_transition_list)
            prev_stage, _ = get_nas_training_stage(epoch-1, args.nas_stage_transition_list)
        else:
            stage = prev_stage = 0
            level = 0

        if prev_stage != stage:
            if args.nas_kd_params:
                if ('teacher_model' not in args.nas_kd_params) or \
                    ('teacher_model' in args.nas_kd_params and
                     args.nas_kd_params['teacher_model'] == 'full_model' and prev_stage == 0):
                    create_nas_kd_policy(model, compression_scheduler, epoch, args.epochs, args)
                elif 'teacher_model' in args.nas_kd_params and \
                     args.nas_kd_params['teacher_model'] == 'prev_stage_model':
                    next_stage_start_epoch = get_next_stage_start_epoch(
                        epoch, args.nas_stage_transition_list, args.epochs)
                    create_nas_kd_policy(model, compression_scheduler, epoch,
                                         next_stage_start_epoch, args)

    # Switch to train mode
    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target_temp) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)

        if args.obj_detection:
            inputs = inputs.to(args.device)
            target = tuple()
            for target_idx in range(len(target_temp[0])):
                temp_list = [elem[target_idx].to(args.device) for elem in target_temp]
                target = target + (temp_list, )
        else:
            inputs, target = inputs.to(args.device), target_temp.to(args.device)

        # Set nas parameters if necessary
        if args.nas:
            if stage == 1:
                ai8x_nas.sample_subnet_kernel(model, level)
            elif stage == 2:
                ai8x_nas.sample_subnet_depth(model, level)
            elif stage == 3:
                ai8x_nas.sample_subnet_width(model, level)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if not hasattr(args, 'kd_policy') or args.kd_policy is None:
            if not hasattr(args, 'nas_kd_policy') or args.nas_kd_policy is None:
                output = model(inputs)
            else:
                output = args.nas_kd_policy.forward(inputs)
        else:
            output = args.kd_policy.forward(inputs)

        if args.out_fold_ratio != 1:
            output = ai8x.unfold_batch(output, args.out_fold_ratio)

        loss = criterion(output, target)
        if not args.obj_detection and not args.dr and not args.kd_relationbased:
            # Measure accuracy if the conditions are set. For `Last Batch` only accuracy
            # calculation last two batches are used as the last batch might include just a few
            # samples.
            if args.show_train_accuracy == 'full' or \
                (args.show_train_accuracy == 'last_batch'
                    and train_step >= len(train_loader)-2):
                if len(output.data.shape) <= 2 or args.regression:
                    classerr.add(output.data, target)
                else:
                    classerr.add(output.data.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2),
                                 target.flatten())
                if not args.regression:
                    acc_stats.append([classerr.value(1),
                                     classerr.value(min(args.num_classes, 5))])
                else:
                    acc_stats.append([classerr.value()])

        # Record loss
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step,
                                                                  steps_per_epoch, loss,
                                                                  optimizer=optimizer,
                                                                  return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        if args.dr:
            loss_optimizer.zero_grad()

        loss.backward()
        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step,
                                                                steps_per_epoch, optimizer)
        optimizer.step()
        if args.dr:
            loss_optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # Reset elastic sampling wrt NAS stage if necessary
        if args.nas:
            if stage == 1:
                ai8x_nas.reset_kernel_sampling(model)
            elif stage == 2:
                ai8x_nas.reset_depth_sampling(model)
            elif stage == 3:
                ai8x_nas.reset_width_sampling(model)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = train_step + 1

        if steps_completed % args.print_freq == 0 or steps_completed == steps_per_epoch:
            # Log some statistics
            errs = OrderedDict()
            if not args.regression:
                if classerr.n != 0:
                    errs['Top1'] = classerr.value(1)
                    if args.num_classes > 5:
                        errs['Top5'] = classerr.value(5)
                else:
                    errs['Top1'] = None
                    errs['Top5'] = None
            else:
                if classerr.n != 0:
                    errs['MSE'] = classerr.value()
                else:
                    errs['MSE'] = None

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            if args.nas:
                stats_dict['NAS-Stage'] = stage
                if stage != 0:
                    stats_dict['NAS-Level'] = level
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Performance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()
    return acc_stats


def update_bn_stats(train_loader, model, args):
    """Routine to update BatchNorm statistics"""
    model.train()
    for (inputs, target) in train_loader:
        inputs, target = inputs.to(args.device), target.to(args.device)
        _ = model(inputs)


def get_all_embeddings(dataset, model):
    """Get all embeddings from the test set"""
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def scaf_test(val_loader, model, accuracy_calculator):
    """Perform test for SCAF"""
    test_embeddings, test_labels = get_all_embeddings(val_loader.dataset, model)
    test_labels = test_labels.squeeze(1)
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, None, None, True
    )
    msglogger.info('Test set accuracy (Precision@1) = %f', accuracies['precision_at_1'])
    return accuracies["precision_at_1"], 0, 0, 0


def validate(val_loader, model, criterion, loggers, args, epoch=-1, tflogger=None):
    """Model validation"""
    assert msglogger is not None
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch, tflogger)


def test(test_loader, model, criterion, loggers, args, mode='ckpt', ckpt_name=None):
    """Model Test"""
    assert msglogger is not None
    if mode == 'ckpt':
        msglogger.info('--- test (ckpt) ---------------------')
        top1, top5, vloss, mAP = _validate(test_loader, model, criterion, loggers, args)
    else:
        msglogger.info('--- test (best) ---------------------')
        if ckpt_name is None:
            best_ckpt_path = os.path.join(msglogger.logdir, 'best.pth.tar')
        else:
            best_ckpt_path = os.path.join(msglogger.logdir, ckpt_name + "_best.pth.tar")
        model = apputils.load_lean_checkpoint(model, best_ckpt_path)
        top1, top5, vloss, mAP = _validate(test_loader, model, criterion, loggers, args)

    return top1, top5, vloss, mAP


def _validate(data_loader, model, criterion, loggers, args, epoch=-1, tflogger=None):
    """Execute the validation/test loop."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    if args.obj_detection:
        map_calculator = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            class_metrics=False,
            iou_thresholds=[0.5],
        ).to(args.device)
        mAP = 0.00
    if not args.regression:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(args.num_classes, 5)))
    else:
        classerr = tnt.MSEMeter()

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = (total_samples + batch_size - 1) // batch_size
    assert msglogger is not None
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    class_probs = []
    class_preds = []
    sample_saved = False  # Track if --save-sample has been done for this validation step

    # Get object detection params
    obj_detection_params = parse_obj_detection_yaml.parse(args.obj_detection_params) \
        if args.obj_detection_params else None

    mAP = 0.0
    have_mAP = False
    with torch.no_grad():
        m = model.module if isinstance(model, DistributedDataParallel) else model

        for validation_step, (inputs, target_temp) in enumerate(data_loader):
            if args.obj_detection:
                if not object_detection_utils.check_target_exists(target_temp):
                    msglogger.info('No target in batch. Ep: %d, '
                                   'validation_step: %d', epoch, validation_step)
                    continue

                filtered_all_images_boxes = None

                inputs = inputs.to(args.device)

                target = tuple()
                for target_idx in range(len(target_temp[0])):
                    temp_list = [elem[target_idx].to(args.device) for elem in target_temp]
                    target = target + (temp_list, )

                # Adjust ground truth index as mAP calculator uses 0-indexed class labels
                labels_list_for_map = [elem[-1].to(args.device) - 1 for elem in target_temp]

                # compute output from model
                output_boxes, output_conf = model(inputs)
                # correct output for accurate loss calculation
                if args.act_mode_8bit:
                    output_boxes /= 128.
                    output_conf /= 128.

                    if (hasattr(m, 'are_locations_wide') and m.are_locations_wide()):
                        output_boxes /= 128.

                    if (hasattr(m, 'are_scores_wide') and m.are_scores_wide()):
                        output_conf /= 128.

                output = (output_boxes, output_conf)

                if target[0]:
                    m, _, _ = model_wrapper.unwrap(model)

                    assert m.detect_objects is not None
                    det_boxes_batch, det_labels_batch, det_scores_batch = \
                        m.detect_objects(output_boxes, output_conf,
                                         min_score=obj_detection_params['nms']['min_score'],
                                         max_overlap=obj_detection_params['nms']['max_overlap'],
                                         top_k=obj_detection_params['nms']['top_k'])

                    # Filter images with only background box
                    filtered_list = list(
                        filter(lambda elem: not (len(elem[1]) == 1 and elem[1][0] == 0),
                               zip(det_boxes_batch, det_labels_batch, det_scores_batch))
                    )

                    # Update mAP Calculator
                    if filtered_list:
                        filtered_all_images_boxes, filtered_all_images_labels, \
                            filtered_all_images_scores = zip(*filtered_list)

                        # mAP calculator uses 0-indexed class labels
                        filtered_all_images_labels = [e - 1 for e in filtered_all_images_labels]

                        # Prepare truths
                        boxes = torch.cat(target[0])
                        labels = torch.cat(labels_list_for_map)

                        gt = [{'boxes': boxes, 'labels': labels}]

                        # Prepare predictions
                        pred_boxes = torch.cat(filtered_all_images_boxes)
                        pred_scores = torch.cat(filtered_all_images_scores)
                        pred_labels = torch.cat(filtered_all_images_labels)

                        preds = [
                            {'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels}
                        ]

                        # Update mAP calculator
                        map_calculator.update(preds=preds, target=gt)
                        have_mAP = True
            else:
                inputs, target = inputs.to(args.device), target_temp.to(args.device)
                # compute output from model
                if args.kd_relationbased:
                    output = args.kd_policy.forward(inputs)
                else:
                    output = model(inputs)
                if args.out_fold_ratio != 1:
                    output = ai8x.unfold_batch(output, args.out_fold_ratio)

                # correct output for accurate loss calculation
                if args.act_mode_8bit:
                    output /= 128.
                    for _, module in model.named_modules():
                        if hasattr(module, 'wide') and module.wide:
                            output /= 128.
                    if args.regression:
                        target /= 128.

            if args.generate_sample is not None and args.act_mode_8bit and not sample_saved:
                sample.generate(args.generate_sample, inputs, target, output,
                                args.dataset, False, args.slice_sample)
                sample_saved = True

            # compute loss
            loss = criterion(output, target)
            if args.kd_relationbased:
                agg_loss = args.kd_policy.before_backward_pass(None, None, None, None,
                                                               loss, None)
                losses[OVERALL_LOSS_KEY].add(agg_loss.overall_loss.item())
            # measure accuracy and record loss
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())

            if not args.obj_detection and not args.kd_relationbased:
                if len(output.data.shape) <= 2 or args.regression:
                    classerr.add(output.data, target)
                else:
                    classerr.add(output.data.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2),
                                 target.flatten())
                if args.display_confusion:
                    confusion.add(output.data, target)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = validation_step + 1
            if steps_completed % args.print_freq == 0 or steps_completed == total_steps:
                if args.display_prcurves and tflogger is not None:
                    # TODO PR Curve generation for Object Detection case is NOT implemented yet
                    class_probs_batch = [nn.functional.softmax(el, dim=0) for el in output]
                    _, class_preds_batch = torch.max(output, 1)
                    class_probs.append(class_probs_batch)
                    class_preds.append(class_preds_batch)
                if args.kd_relationbased:
                    stats = (
                        '',
                        OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
                                     ('Overall Loss', losses[OVERALL_LOSS_KEY].mean)])
                    )

                elif args.obj_detection:
                    # Only run compute() if there is at least one new update()
                    if have_mAP:
                        mAP = map_calculator.compute()['map_50']
                        have_mAP = False
                    stats = (
                        '',
                        OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
                                     ('mAP', mAP)])
                    )
                elif args.regression:
                    stats = (
                        '',
                        OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
                                    ('MSE', classerr.value())])
                    )
                else:
                    stats = (
                        '',
                        OrderedDict([('Loss', losses[OBJECTIVE_LOSS_KEY].mean),
                                    ('Top1', classerr.value(1))])
                    )
                    if args.num_classes > 5:
                        stats[1]['Top5'] = classerr.value(5)

                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)

                if args.display_prcurves and tflogger is not None:
                    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
                    test_preds = torch.cat(class_preds)
                    for i in range(args.num_classes):
                        tb_preds = test_preds == i
                        tb_probs = test_probs[:, i]
                        tflogger.tblogger.writer.add_pr_curve(str(args.labels[i]), tb_preds,
                                                              tb_probs, global_step=epoch)

                if args.display_embedding and tflogger is not None \
                   and steps_completed == total_steps:
                    def select_n_random(data, labels, features, n=100):
                        """Selects n random datapoints, their corresponding labels and features"""
                        assert len(data) == len(labels) == len(features)

                        perm = torch.randperm(len(data))
                        return data[perm][:n], labels[perm][:n], features[perm][:n]

                    # Select up to 100 random images and their target indices
                    images, labels, features = select_n_random(inputs, target, output,
                                                               n=min(100, len(inputs)))

                    # Get the class labels for each image
                    class_labels = [args.labels[lab] for lab in labels]

                    tflogger.tblogger.writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=args.visualize_fn(images, args),
                        global_step=epoch,
                        tag='verification/embedding'
                    )

    if args.obj_detection:
        msglogger.info('==> mAP: %.5f    Loss: %.3f\n', mAP,
                       losses[OBJECTIVE_LOSS_KEY].mean)
        return 0, 0, losses[OBJECTIVE_LOSS_KEY].mean, mAP

    if args.kd_relationbased:
        msglogger.info('==> Overall Loss: %.3f\n',
                       losses[OVERALL_LOSS_KEY].mean)
        return 0, 0, losses[OVERALL_LOSS_KEY].mean, 0

    if args.regression:
        msglogger.info('==> MSE: %.5f    Loss: %.3f\n',
                       classerr.value(), losses[OBJECTIVE_LOSS_KEY].mean)
        return classerr.value(), .0, losses[OBJECTIVE_LOSS_KEY].mean, 0

    if args.num_classes > 5:
        msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                       classerr.value()[0], classerr.value()[1],
                       losses[OBJECTIVE_LOSS_KEY].mean)
    else:
        msglogger.info('==> Top1: %.3f    Loss: %.3f\n',
                       classerr.value()[0], losses[OBJECTIVE_LOSS_KEY].mean)

    if args.display_confusion:
        msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
        if tflogger is not None:
            cf = nnplot.confusion_matrix(confusion.value(), args.labels)
            tflogger.tblogger.writer.add_image('Validation/ConfusionMatrix', cf, epoch,
                                               dataformats='HWC')
    return classerr.value(1), classerr.value(min(args.num_classes, 5)), \
        losses[OBJECTIVE_LOSS_KEY].mean, 0


def update_training_scores_history(perf_scores_history, model, top1, top5, mAP, vloss, epoch,
                                   args):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    _, _, params_nnz_cnt = distiller.model_params_stats(model, param_dims=[2, 3, 4])

    perf_scores_history.append(
        distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                     'top1': top1, 'top5': top5, 'mAP': mAP, 'vloss': -vloss,
                                     'epoch': epoch}))

    assert msglogger is not None
    if args.kd_relationbased:
        # Keep perf_scores_history sorted from best to worst based on overall loss
        # overall_loss = student_loss*student_weight + distillation_loss*distillation_weight
        perf_scores_history.sort(key=operator.attrgetter('params_nnz_cnt', 'vloss', 'epoch'),
                                 reverse=True)
        for score in perf_scores_history[:args.num_best_scores]:
            msglogger.info('==> Best [Overall Loss: %f on epoch: %d]',
                           -score.vloss, score.epoch)

    elif args.obj_detection:
        # Keep perf_scores_history sorted from best to worst
        # Sort by mAP as main sort key, then sort by vloss and epoch
        perf_scores_history.sort(key=operator.attrgetter('mAP', 'vloss', 'epoch'),
                                 reverse=True)
        for score in perf_scores_history[:args.num_best_scores]:

            msglogger.info('==> Best [mAP: %f   vloss: %f   '
                           'Params: %d on epoch: %d]',
                           score.mAP, -score.vloss, -score.params_nnz_cnt,
                           score.epoch)
    elif args.regression:
        # Sort by MSE as main sort key, then sort by epoch
        perf_scores_history.sort(key=operator.attrgetter('epoch'), reverse=True)
        perf_scores_history.sort(key=operator.attrgetter('top1'), reverse=False)

        for score in perf_scores_history[:args.num_best_scores]:
            msglogger.info('==> Best [Top 1 (MSE): %.5f   '
                           'Params: %d on epoch: %d]',
                           score.top1, -score.params_nnz_cnt,
                           score.epoch)

    else:
        # Keep perf_scores_history sorted from best to worst
        # Sort by top1 as main sort key, then sort by top5 and epoch
        perf_scores_history.sort(key=operator.attrgetter('top1', 'top5', 'epoch'),
                                 reverse=True)
        for score in perf_scores_history[:args.num_best_scores]:
            if args.num_classes > 5:
                msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   '
                               'Params: %d on epoch: %d]',
                               score.top1, score.top5, -score.params_nnz_cnt,
                               score.epoch)
            else:
                msglogger.info('==> Best [Top1: %.3f   '
                               'Params: %d on epoch: %d]',
                               score.top1, -score.params_nnz_cnt,
                               score.epoch)


def summarize_model(model, dataset, which_summary, filename='model'):
    """summarize_model"""
    if which_summary.startswith('png'):
        if which_summary == 'png_simplified':
            ai8x.onnx_export_prep(model, simplify=True, remove_clamp=True)
        model_summaries.draw_img_classifier_to_file(model, filename + '.png', dataset,
                                                    which_summary == 'png_w_params')
    elif which_summary in ['onnx', 'onnx_simplified']:
        ai8x.onnx_export_prep(model, simplify=which_summary == 'onnx_simplified')
        model_summaries.export_img_classifier_to_onnx(
            model,
            filename + '.onnx',
            dataset,
            add_softmax=False,
        )
    else:
        distiller.model_summary(model, which_summary, dataset, log_1d=True)


def create_nas_training_stage_list(model, nas_policy):
    """Create list to define NAS stage transition epochs"""
    stage_transition_list = []
    assert msglogger is not None
    msglogger.info('NAS Policy: %s', nas_policy)

    stage_transition_list.append((nas_policy['start_epoch'], 0, 0))

    max_kernel_level = model.get_max_elastic_kernel_level()
    if nas_policy['elastic_kernel']['leveling']:
        for level in range(max_kernel_level):
            stage_transition_list.append((stage_transition_list[-1][0] +
                                          nas_policy['elastic_kernel']['num_epochs'], 1, level+1))
    else:
        stage_transition_list.append((stage_transition_list[-1][0] +
                                      nas_policy['elastic_kernel']['num_epochs'], 1,
                                      max_kernel_level))

    max_depth_level = model.get_max_elastic_depth_level()
    if nas_policy['elastic_depth']['leveling']:
        for level in range(max_depth_level):
            stage_transition_list.append((stage_transition_list[-1][0] +
                                          nas_policy['elastic_depth']['num_epochs'], 2, level+1))
    else:
        stage_transition_list.append((stage_transition_list[-1][0] +
                                      nas_policy['elastic_depth']['num_epochs'], 2,
                                      max_depth_level))

    max_width_level = model.get_max_elastic_width_level()
    if nas_policy['elastic_width']['leveling']:
        for level in range(max_width_level):
            stage_transition_list.append((stage_transition_list[-1][0] +
                                          nas_policy['elastic_width']['num_epochs'], 3, level+1))
    else:
        stage_transition_list.append((stage_transition_list[-1][0] +
                                      nas_policy['elastic_width']['num_epochs'], 3,
                                      max_width_level))

    return stage_transition_list


def get_nas_training_stage(epoch, stage_transition_list):
    """Returns current stage of NAS"""
    for t in stage_transition_list:
        if epoch < t[0]:
            break

    return t[1], t[2]  # pylint: disable=undefined-loop-variable


def get_next_stage_start_epoch(epoch, stage_transition_list, num_epochs):
    """Returns the starting epoch of the following stage"""
    current_stage = None
    for stg_idx, t in enumerate(stage_transition_list):
        if epoch < t[0]:
            if current_stage is None:
                current_stage = t[1]
            if stg_idx == len(stage_transition_list)-1:
                return num_epochs
            if current_stage != stage_transition_list[stg_idx+1][1]:
                return t[0]
    return num_epochs


class missingdict(dict):
    """This is a little trick to prevent KeyError"""

    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def update_old_model_params(model_path, model_new):
    """Adds missing model parameters added with default values.
    This is mainly due to the saved checkpoints from previous versions of the repo.
    New model is saved to `model_path` and the old model copied into the same file_path with
    `__obsolete__` prefix."""
    is_model_old = False
    model_old = torch.load(model_path,
                           map_location=lambda storage, loc: storage)
    # Fix up any instances of DataParallel
    old_dict = model_old['state_dict'].copy()
    for k in old_dict:
        if k.startswith('module.'):
            model_old['state_dict'][k[7:]] = old_dict[k]
    for new_key, new_val in model_new.state_dict().items():
        if new_key not in model_old['state_dict'] and '.bn.' not in new_key:
            is_model_old = True
            model_old['state_dict'][new_key] = new_val
            if 'compression_sched' in model_old:
                if 'masks_dict' in model_old['compression_sched']:
                    model_old['compression_sched']['masks_dict'][new_key] = None

    if is_model_old:
        dir_path, file_name = os.path.split(model_path)
        new_file_name = '__obsolete__' + file_name
        old_model_path = os.path.join(dir_path, new_file_name)
        os.rename(model_path, old_model_path)
        torch.save(model_old, model_path)
        msglogger.info('Model `%s` is old. Missing parameters added with default values!',
                       model_path)


if __name__ == '__main__':
    try:
        np.set_printoptions(threshold=sys.maxsize, linewidth=190)
        torch.set_printoptions(threshold=sys.maxsize, linewidth=190)
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in
            # stdout - once from the logging operation and once from re-raising the exception.
            # So we remove the stdout logging handler before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers
                                  if not isinstance(h, logging.StreamHandler)]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: %s', os.path.realpath(msglogger.log_filename))
