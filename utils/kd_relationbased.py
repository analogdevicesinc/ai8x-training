#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2023 Analog Devices, Inc.
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
""" Relation based Knowledge Distillation Policy"""

from collections import namedtuple

import torch
from torch import nn

from distiller.policy import LossComponent, PolicyLoss, ScheduledTrainingPolicy

DistillationLossWeights = namedtuple('DistillationLossWeights',
                                     ['distill', 'student', 'teacher'])


class RelationBasedKDPolicy(ScheduledTrainingPolicy):
    """
    Relation based Knowledge Distillation Policy class based on
    the distiller's ScheduledTrainingPolicy class.
    """
    def __init__(self, student_model, teacher_model,
                 loss_weights=DistillationLossWeights(0.5, 0.5, 0), act_mode_8bit=False):
        super().__init__()

        self.student = student_model
        self.teacher = teacher_model
        self.teacher_output = None
        self.student_output = None
        self.loss_wts = loss_weights
        self.distillation_loss = nn.MSELoss()
        self.overall_loss = None
        self.act_mode_8bit = act_mode_8bit

        # Active is always true, because test will be based on the overall loss and it will be
        # realized outside of the epoch loop
        self.active = True

    def forward(self, *inputs):
        """
        Performs forward propagation through both student and teacher models and
        caches the outputs.This function MUST be used instead of calling the student
        model directly.

        Returns:
            The student model's returned output, to be consistent with what a
            script using this would expect
        """
        if not self.active:
            return self.student(*inputs)

        with torch.no_grad():
            self.teacher_output = self.teacher(*inputs)

        out = self.student(*inputs)
        if self.act_mode_8bit:
            out /= 128.
        self.student_output = out.clone()

        return out

    # pylint: disable=unused-argument
    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        """
        Not used
        """

    # pylint: disable=unused-argument
    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        """
        Not used
        """

    # pylint: disable=unused-argument
    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss,
                             zeros_mask_dict, optimizer=None):
        """
        Returns the overall loss, which is a weighted sum of the student loss and
        the distillation loss
        """

        if self.student_output is None or self.teacher_output is None:
            raise RuntimeError("KnowledgeDistillationPolicy: Student and or teacher outputs"
                               "were not cached. Make sure to call "
                               "KnowledgeDistillationPolicy.forward() in your script instead of "
                               "calling the model directly.")

        distillation_loss = self.distillation_loss(self.student_output, self.teacher_output)

        overall_loss = self.loss_wts.student * loss + self.loss_wts.distill * distillation_loss

        # For logging purposes, we return the un-scaled distillation loss so it's
        # comparable between runs with different temperatures
        return PolicyLoss(overall_loss,
                          [LossComponent('Distill Loss', distillation_loss)])
