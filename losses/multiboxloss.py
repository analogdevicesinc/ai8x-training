###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
# Multi Box Loss code is from GitHub repo:
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection:
# MIT License
# Copyright (c) 2019 Sagar Vinodababu
# Codes slightly with few minor modifications
"""
    Multi-box Loss for Object Detection Models
"""
import torch
from torch import nn

import utils.object_detection_utils as obj_det_utils


class MultiBoxLoss(nn.Module):
    """
        Multi-box Loss for Object Detection Models

        Multi-box loss is a weighted sum of the loss components: location loss and confidence loss.

        Location loss  measures the distance between the annotated object boxes and the predicted
        bounding boxes. SSD paper uses smooth L1 loss for this purpose and so as the below
        implementation.

        Confidence Loss measures the performance of the  object classification per box.Categorical
        cross-entropy is used to compute this loss.

        SSD paper (Liu, W. et al. (2016). SSD: Single Shot MultiBox Detector,
        https://doi.org/10.1007/978-3-319-46448-0_2)
    """
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., device='cpu'):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = obj_det_utils.cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.device = device

    def forward(self, output, target):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the prior boxes
        :param predicted_scores: class scores for each of the encoded locations/boxes
        :param boxes: true object bounding boxes in boundary coordinates
        :param labels: true object labels
        :return: multibox loss
        """

        predicted_locs, predicted_scores = output
        boxes, labels = target

        shape_1, shape_2 = predicted_locs.shape[1:]
        if shape_2 > shape_1:
            predicted_locs = torch.transpose(predicted_locs, 1, 2)

        shape_1, shape_2 = predicted_scores.shape[1:]
        if shape_2 > shape_1:
            predicted_scores = torch.transpose(predicted_scores, 1, 2)

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            if n_objects > 0:
                overlap = obj_det_utils.find_jaccard_overlap(boxes[i], self.priors_xy)

                # For each prior, find the object that has the maximum overlap
                overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

                # We don't want a situation where an object is not represented in our positive
                # (non-background) priors -
                # 1. An object might not be the best object for all priors, and is therefore not in
                #    object_for_each_prior.
                # 2. All priors with the object may be assigned as background based on the
                #    threshold (0.5).

                # To remedy this -
                # First, find the prior that has the maximum overlap for each object.
                _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

                # Then, assign each object to the corresponding maximum-overlap-prior.
                # (This fixes 1.)
                object_for_each_prior[prior_for_each_object] = \
                    torch.LongTensor(range(n_objects)).to(self.device)

                # To ensure these priors qualify, artificially give them an overlap of greater
                # than 0.5. (This fixes 2.)
                overlap_for_each_prior[prior_for_each_object] = 1.

                # Labels for each prior
                label_for_each_prior = labels[i][object_for_each_prior]
                # Set priors whose overlaps with objects are less than the threshold to be
                # background (no object)
                label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

                # Store
                true_classes[i] = label_for_each_prior

                # Encode center-size object coordinates into the form we regressed predicted boxes
                true_locs[i] = obj_det_utils.cxcy_to_gcxgcy(
                                   obj_det_utils.xy_to_cxcy(boxes[i][object_for_each_prior]),
                                   self.priors_cxcy)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is
        # across multiple dimensions (N & number_of_priors)
        # So, if predicted_locs has the shape (N, number_of_priors, 4),
        # predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest)
        # negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there
        # is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image,
        # and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.contiguous().view(-1, n_classes),
                                           true_classes.view(-1))  # (N * number_of_priors)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, number_of_priors)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take
        # top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, number_of_priors)
        conf_loss_neg[positive_priors] = 0.  # (N, number_of_priors), positive priors are ignored
        # (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, number_of_priors),
        # sorted by decreasing hardness
        hardness_ranks = \
            torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)
        # (N, number_of_priors)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        # (N, number_of_priors)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive
        # and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()
        # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
