#
# Copyright (c) 2018 Intel Corporation
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

"""Loggers frontends and backends.

- TensorBoardLogger logs to files that can be read by Google's TensorBoard.

Note that not all loggers implement all logging methods.
"""

import torch
from torch.utils.tensorboard import SummaryWriter

import distiller
# pylint: disable=no-name-in-module
from distiller.data_loggers.logger import DataLogger
# pylint: enable=no-name-in-module
from distiller.utils import density, norm_filters, sparsity, sparsity_2D, to_np


class TensorBoardLogger(DataLogger):
    """
    TensorBoardLogger
    """
    def __init__(self, logdir, comment=''):
        super(TensorBoardLogger, self).__init__()
        # Set the tensorboard logger
        self.writer = SummaryWriter(logdir, comment=comment)
        print('\n--------------------------------------------------------')
        print('Logging to TensorBoard - remember to execute the server:')
        print('> tensorboard --logdir=\'./logs\'\n')

        # Hard-code these preferences for now
        self.log_gradients = False  # True
        self.logged_params = ['weight']  # ['weight', 'bias']

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        def total_steps(total, epoch, completed):
            return total*epoch + completed

        prefix = stats_dict[0]
        stats_dict = stats_dict[1]

        for tag, value in stats_dict.items():
            self.writer.add_scalar(prefix+tag.replace(' ', '_'), value,
                                   total_steps(total, epoch, completed))

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        group = stat_name + '/activations/' + phase + "/"
        for tag, value in activation_stats.items():
            self.writer.add_scalar(group+tag, value, epoch)

    def log_weights_sparsity(self, model, epoch):
        params_size = 0
        sparse_params_size = 0

        for name, param in model.state_dict().items():
            if param.dim() in [2, 4]:
                _density = density(param)
                params_size += torch.numel(param)
                sparse_params_size += param.numel() * _density
                self.writer.add_scalar('sparsity/weights/' + name,
                                       sparsity(param)*100, epoch)
                self.writer.add_scalar('sparsity-2D/weights/' + name,
                                       sparsity_2D(param)*100, epoch)

        self.writer.add_scalar("sparsity/weights/total",
                               100*(1 - sparse_params_size/params_size), epoch)

    def log_weights_filter_magnitude(
            self,
            model,
            epoch,
            multi_graphs=False,  # pylint: disable=unused-argument
    ):
        """Log the L1-magnitude of the weights tensors.
        """
        for name, param in model.state_dict().items():
            if param.dim() in [4]:
                self.writer.add_scalars('magnitude/filters/' + name,
                                        list(to_np(norm_filters(param))), epoch)

    def log_weights_distribution(self, named_params, steps_completed):
        if named_params is None:
            return
        for tag, value in named_params:
            tag = tag.replace('.', '/')
            if any(substring in tag for substring in self.logged_params):
                self.writer.add_histogram(tag, to_np(value), steps_completed)
            if self.log_gradients:
                self.writer.add_histogram(tag+'/grad', to_np(value.grad), steps_completed)

    def log_model_buffers(self, model, buffer_names, tag_prefix, epoch, completed, total, freq):
        """Logs values of model buffers.

        Notes:
            1. Buffers are logged separately per-layer (i.e. module) within model
            2. All values in a single buffer are logged such that they will be displayed on the
               same graph in TensorBoard
            3. Similarly, if multiple buffers are provided in buffer_names, all are presented on
               the same graph.
               If this is un-desirable, call the function separately for each buffer
            4. USE WITH CAUTION: While sometimes desirable, displaying multiple distinct values in
               a single graph isn't well supported in TensorBoard. It is achieved using a
               work-around, which slows down TensorBoard loading time considerably as the number
               of distinct values increases.
               Therefore, while not limited, this function is only meant for use with a very
               limited number of buffers and/or values, e.g. 2-5.

        """
        for module_name, module in model.named_modules():
            if distiller.has_children(module):
                continue

            sd = module.state_dict()
            values = []
            for buf_name in buffer_names:
                try:
                    values += sd[buf_name].view(-1).tolist()
                except KeyError:
                    continue

            if values:
                tag = '/'.join([tag_prefix, module_name])
                self.writer.add_scalars(tag, values, total * epoch + completed)
