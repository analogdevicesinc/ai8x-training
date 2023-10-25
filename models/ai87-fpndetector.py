###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
FPN (Feature Pyramid Network) Object Detection Model
"""
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

import ai8x
import utils.object_detection_utils as obj_detect_utils


# TODO: May move to ai8x_blocks
class Residual(nn.Module):
    """
    Residual connection module
    """
    def __init__(self, in_channels, out_channels, res_layer_count=2, preprocess_kernel_size=3,
                 batchnorm='NoAffine', pooling=False, bias=True,
                 remove_residual=False, **kwargs):
        super().__init__()
        if preprocess_kernel_size == 3:
            padding = 1
        elif preprocess_kernel_size == 1:
            padding = 0
        else:
            raise ValueError('Preprocess kernel size could be 3 or 1')

        if pooling:
            self.preprocess_layer = \
                ai8x.FusedMaxPoolConv2dBNReLU(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=preprocess_kernel_size,
                                              padding=padding,
                                              batchnorm=batchnorm,
                                              bias=bias, **kwargs)
        else:
            self.preprocess_layer = \
                ai8x.FusedConv2dBNReLU(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=preprocess_kernel_size,
                                       padding=padding,
                                       batchnorm=batchnorm,
                                       bias=bias, **kwargs)

        self.res_layers = nn.ModuleDict()

        for index in range(res_layer_count):
            layer_name = f'conv_{index}'
            self.res_layers[layer_name] = \
                ai8x.FusedConv2dBNReLU(in_channels=out_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       padding=1,
                                       batchnorm=batchnorm,
                                       bias=bias, **kwargs)

        self.add = ai8x.Add()
        self.remove_residual = remove_residual

    def forward(self, input_data):
        """
        Forward propagation for the residual block
        """
        x = self.preprocess_layer(input_data)
        x1 = x
        for res_layer in self.res_layers:
            x = self.res_layers[res_layer](x)
        if not self.remove_residual:
            x = self.add(x, x1)
        return x


class ResNetBackbone(nn.Module):
    """
    ResNet Backbone module
    """
    def __init__(self, in_channels):
        super().__init__()
        self.residual_256_320 = Residual(in_channels, 64, pooling=True, remove_residual=True)
        self.residual_128_160 = Residual(64, 64, pooling=True)

        self.residual_64_80_pre = Residual(64, 64, pooling=False)
        self.residual_64_80 = Residual(64, 64, pooling=True)

        self.residual_32_40 = Residual(64, 64, res_layer_count=2, pooling=True)
        self.residual_16_20 = Residual(64, 128, res_layer_count=2, pooling=True)
        self.residual_8_10 = Residual(128, 128, res_layer_count=2, pooling=True)

    def forward(self, input_data):
        """
        Forward propagation for the ResNet Backbone
        """
        prep0 = self.residual_256_320(input_data)

        enc_64_80 = self.residual_128_160(prep0)
        enc_64_80_2 = self.residual_64_80_pre(enc_64_80)

        enc_32_40 = self.residual_64_80(enc_64_80_2)
        enc_16_20 = self.residual_32_40(enc_32_40)
        enc_8_10 = self.residual_16_20(enc_16_20)
        enc_4_5 = self.residual_8_10(enc_8_10)

        return [enc_32_40, enc_16_20, enc_8_10, enc_4_5]


class FPN(nn.Module):
    """
    FPN: Feature Pyramid Network
    """
    def __init__(self, channels_32_40, channels_16_20, channels_8_10, channels_4_5,
                 batchnorm='NoAffine', bias=True,  **kwargs):
        super().__init__()

        default_ch_count = 64

        self.skip_32_40 = ai8x.FusedConv2dBNReLU(in_channels=channels_32_40,
                                                 out_channels=channels_32_40,
                                                 kernel_size=1,
                                                 padding=0,
                                                 batchnorm=batchnorm,
                                                 bias=bias, **kwargs)

        self.skip_16_20 = ai8x.FusedConv2dBNReLU(in_channels=channels_16_20,
                                                 out_channels=default_ch_count,
                                                 kernel_size=1,
                                                 padding=0,
                                                 batchnorm=batchnorm,
                                                 bias=bias, **kwargs)

        self.skip_8_10 = ai8x.FusedConv2dBNReLU(in_channels=channels_8_10,
                                                out_channels=default_ch_count,
                                                kernel_size=1,
                                                padding=0,
                                                batchnorm=batchnorm,
                                                bias=bias, **kwargs)

        self.skip_4_5 = ai8x.FusedConv2dBNReLU(in_channels=channels_4_5,
                                               out_channels=default_ch_count,
                                               kernel_size=1,
                                               padding=0,
                                               batchnorm=batchnorm,
                                               bias=bias, **kwargs)

        self.upconv_4_5 = ai8x.ConvTranspose2d(
            default_ch_count, default_ch_count, 3, stride=2, padding=1)

        self.process_8_10 = ai8x.FusedConv2dBNReLU(in_channels=default_ch_count,
                                                   out_channels=default_ch_count,
                                                   kernel_size=3,
                                                   padding=1,
                                                   batchnorm=batchnorm,
                                                   bias=bias, **kwargs)

        self.upconv_8_10 = ai8x.ConvTranspose2d(
            default_ch_count, default_ch_count, 3, stride=2, padding=1)

        self.process_16_20 = ai8x.FusedConv2dBNReLU(in_channels=default_ch_count,
                                                    out_channels=default_ch_count,
                                                    kernel_size=3,
                                                    padding=1,
                                                    batchnorm=batchnorm,
                                                    bias=bias, **kwargs)

        self.upconv_16_20 = ai8x.ConvTranspose2d(
            default_ch_count, default_ch_count, 3, stride=2, padding=1)

        self.process_32_40 = ai8x.FusedConv2dBNReLU(in_channels=default_ch_count,
                                                    out_channels=default_ch_count,
                                                    kernel_size=3,
                                                    padding=1,
                                                    batchnorm=batchnorm,
                                                    bias=bias, **kwargs)

        self.add = ai8x.Add()

    def forward(self, input_32_40, input_16_20, input_8_10, input_4_5):
        """
        Forward propagation for the FPN: Feature Pyramid Network
        """
        skip_32_40 = self.skip_32_40(input_32_40)
        skip_16_20 = self.skip_16_20(input_16_20)
        skip_8_10 = self.skip_8_10(input_8_10)
        skip_4_5 = self.skip_4_5(input_4_5)

        out_4_5 = skip_4_5
        out_8_10 = self.add(skip_8_10, self.upconv_4_5(out_4_5))
        out_8_10 = self.process_8_10(out_8_10)
        out_16_20 = self.add(skip_16_20, self.upconv_8_10(out_8_10))
        out_16_20 = self.process_16_20(out_16_20)
        out_32_40 = self.add(skip_32_40, self.upconv_16_20(out_16_20))
        out_32_40 = self.process_32_40(out_32_40)

        return [out_32_40, out_16_20, out_8_10, out_4_5]


class ClassificationModel(nn.Module):
    """
    Classification module for class scores
    """
    def __init__(self, num_anchors=6, num_classes=21, feature_size=64,
                 bias=True,  wide=False, **kwargs):
        super().__init__()

        self.res_conv1 = Residual(in_channels=feature_size,
                                  out_channels=feature_size,
                                  pooling=False)

        self.res_conv2 = Residual(in_channels=feature_size,
                                  out_channels=feature_size,
                                  pooling=False)

        self.conv5 = ai8x.Conv2d(in_channels=feature_size,
                                 out_channels=num_anchors*num_classes,
                                 kernel_size=3,
                                 padding=1,
                                 bias=bias,
                                 wide=wide,
                                 **kwargs)

        self.sigmoid = nn.Sigmoid()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, input_data):
        """
        Forward propagation for the Classification Module
        """
        out = self.res_conv1(input_data)
        out = self.res_conv2(out)
        out = self.conv5(out)
        out = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out.shape
        out = out.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out.contiguous().view(input_data.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):
    """
    Classification module for box regression outputs
    """
    def __init__(self, num_anchors=6, feature_size=64,
                 bias=True, wide=False, **kwargs):

        super().__init__()

        self.res_conv1 = Residual(in_channels=feature_size,
                                  out_channels=feature_size,
                                  pooling=False)

        self.res_conv2 = Residual(in_channels=feature_size,
                                  out_channels=feature_size,
                                  pooling=False)

        self.conv5 = ai8x.Conv2d(in_channels=feature_size,
                                 out_channels=num_anchors*4,
                                 kernel_size=3,
                                 padding=1,
                                 bias=bias, wide=wide, **kwargs)

    def forward(self, input_data):
        """
        Forward propagation for the Regression Module
        """
        out = self.res_conv1(input_data)
        out = self.res_conv2(out)
        out = self.conv5(out)

        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(input_data.shape[0], -1, 4)


class FeaturePyramidNetworkDetector(nn.Module):
    """
    The FeaturePyramidNetworkDetector network consisting
        * ResNet Backbone Network
        * Feature Pyramid Network
        * Classification Network
        * Regression Network
    """

    def __init__(self, num_classes=21,
                 num_channels=3,  # pylint: disable=unused-argument
                 dimensions=(256, 320),  # pylint: disable=unused-argument
                 in_channels=3,
                 preprocess_channels=64,
                 channels_32_40=64,
                 channels_16_20=64, channels_8_10=128,
                 channels_4_5=128,
                 device='cpu',
                 wide_locations=False,
                 wide_scores=False,
                 **kwargs):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.preprocess_layer_1 = ai8x.FusedConv2dBNReLU(in_channels=in_channels,
                                                         out_channels=preprocess_channels,
                                                         kernel_size=3,
                                                         padding=1,
                                                         bias=True)

        self.preprocess_layer_2 = ai8x.FusedConv2dBNReLU(in_channels=preprocess_channels,
                                                         out_channels=preprocess_channels,
                                                         kernel_size=3,
                                                         padding=1,
                                                         bias=True)

        self.backbone = ResNetBackbone(preprocess_channels)
        self.fpn = FPN(channels_32_40, channels_16_20, channels_8_10, channels_4_5, **kwargs)
        self.classification_net = ClassificationModel(
            num_classes=self.num_classes, feature_size=64, wide=wide_scores, **kwargs)
        self.regression_net = RegressionModel(feature_size=64, wide=wide_locations, **kwargs)

        self.priors_cxcy = self.__class__.create_prior_boxes(self.device)

    def forward(self, input_data):
        """
        The FeaturePyramidNetworkDetector forward propagation
            * Runs preprocessing layers and ResNet Backbone Network
            * Runs Feature Pyramid Network
            * Runs Classification Network for all features
            * Runs Regression Network for all features
        """

        x = self.preprocess_layer_1(input_data)
        x = self.preprocess_layer_2(x)

        backbone_features = self.backbone(x)

        pyramide_features = self.fpn(*backbone_features)

        regression = \
            torch.cat([self.regression_net(feature) for feature in pyramide_features],
                      dim=1)
        classification = \
            torch.cat([self.classification_net(feature) for feature in pyramide_features],
                      dim=1)

        return regression, classification

    @staticmethod
    def create_prior_boxes(device='cpu'):
        """
        Create the prior (default) boxes
        :return: prior boxes in center-size coordinates
        """

        fmap_dims = {'f0': (32, 40),
                     'f1': (16, 20),
                     'f2': (8, 10),
                     'f3': (4, 5)}

        fmap_dim_scales = {'f0': 0.1,
                           'f1': 0.2,
                           'f2': 0.4,
                           'f3': 0.8}

        fmaps = list(fmap_dims.keys())

        obj_scales = {'s0': 2 ** 0,
                      's1': 2 ** (1.5 / 3.0),
                      }

        aspect_ratios = {'ar0': 0.5,
                         'ar1': 1,
                         'ar2': 2
                         }

        prior_boxes = []

        for fmap in fmaps:
            for i in range(fmap_dims[fmap][0]):
                for j in range(fmap_dims[fmap][1]):
                    cx = (j + 0.5) / fmap_dims[fmap][1]
                    cy = (i + 0.5) / fmap_dims[fmap][0]

                    for ratio in aspect_ratios.values():
                        for obj_scale in obj_scales.values():

                            prior_boxes.append([cx, cy,
                                                (obj_scale*fmap_dim_scales[fmap]) * sqrt(ratio),
                                                (obj_scale*fmap_dim_scales[fmap]) / sqrt(ratio)])

        prior_boxes = torch.tensor(prior_boxes, dtype=torch.float, device=device)
        prior_boxes.clamp_(0, 1)  # (num_priors, 4)

        return prior_boxes

    # TODO: Migrate detect_objects to obj_detection_utils: prior boxes should also be parametrized
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the locations and class scores to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum
        threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the prior boxes, a tensor of
        dimensions
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of
        dimensions
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score
        is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the
        top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = obj_detect_utils.cxcy_to_xy(
                obj_detect_utils.gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

            # Lists to store boxes and scores for this image
            image_boxes = []
            image_labels = []
            image_scores = []

            # Check for each class
            for c in range(1, self.num_classes):
                # Keep only predicted boxes and scores where scores for this class are above the
                # minimum score
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = obj_detect_utils.find_jaccard_overlap(class_decoded_locs,
                                                                class_decoded_locs)
                # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.bool tensor to keep track of which predicted boxes to suppress
                # True implies suppress, False implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool, device=self.device)
                # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box]:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum
                    # overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.logical_or(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = False

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~suppress])
                image_labels.append(
                    torch.tensor((~suppress).sum().item() * [c],
                                 dtype=torch.long, device=self.device))
                image_scores.append(class_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.tensor([[0., 0., 1., 1.]],
                                                dtype=torch.float, device=self.device))
                image_labels.append(torch.tensor([0], dtype=torch.long, device=self.device))
                image_scores.append(torch.tensor([0.], dtype=torch.float, device=self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

    def are_locations_wide(self):
        """
        Returns whether model uses wide outputs for the regression part (box locations)
        """
        return self.regression_net.conv5.wide

    def are_scores_wide(self):
        """
        Returns whether model uses wide outputs for the classification part (box predictions)
        """
        return self.classification_net.conv5.wide


def ai87fpndetector(pretrained=False, **kwargs):
    """
    Constructs a Feature Pyramid Network Detector model
    """
    assert not pretrained
    return FeaturePyramidNetworkDetector(wide_locations=False, wide_scores=True, **kwargs)


models = [
    {
        'name': 'ai87fpndetector',
        'min_input': 1,
        'dim': 2,
    }
]
