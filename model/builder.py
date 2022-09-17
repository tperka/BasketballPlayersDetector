import math
from time import perf_counter

import torch
import torchvision
from torch import nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model import basenet_configs
from model.BasketballDetector import BasketballDetector
from model.modules import FPN, ConvModule, MBConvN, ResBlock



def scale_width(w, w_factor):
    w *= w_factor
    new_w = (int(w + 4) // 8) * 8
    new_w = max(8, new_w)
    if new_w < 0.9 * w:
        new_w += 8
    return int(new_w)


def make_bm_modules(cfg, batch_norm=True):
    # Each module is a list of sequential layers operating at the same spacial dimension followed by MaxPool2d
    modules = nn.ModuleList()
    # Number of output channels in each module
    out_channels = []
    layers = []
    def flush():
        nonlocal layers
        modules.append(nn.Sequential(*layers))
        out_channels.append(in_channels)
        layers = []

    in_channels = 3
    w_factor = cfg["w_factor"]
    scaled_widths = [(scale_width(w_in, w_factor), scale_width(w_out, w_factor)) for w_in, w_out in cfg['widths']]
    scaled_depths = [math.ceil(depth * cfg["d_factor"]) for depth in cfg["depths"]]

    layers = [ConvModule(in_channels, scaled_widths[0][0], stride=2, padding=1)]
    flush()
    for width, depth, kernel_size, stride, p, r, expansion in zip(scaled_widths, scaled_depths, cfg["kernel_sizes"], cfg["strides"], cfg["ps"], cfg["rs"], cfg["expansion_factors"]):
        w_in, w_out = width
        # NOTE: Only first layer has stride
        layers.append(MBConvN(w_in, w_out, expansion, kernel_size=kernel_size, stride=stride, r=r, p=p))
        layers += [MBConvN(w_out, w_out, expansion, kernel_size=kernel_size, stride=1, r=r, p=p)] * (depth - 1)
        in_channels = w_out
        if stride > 1:
            flush()

    end_lay = scaled_widths[-1][-1]
    layers += [ConvModule(end_lay, end_lay, kernel_size=1), nn.AdaptiveAvgPool2d((None, None))]
    return modules, out_channels


def make_vgg_modules(cfg, batch_norm=False, activation=nn.ReLU, channel_attention=False):
    modules = nn.ModuleList()
    out_channels = []

    # RGB image has 3 channels
    in_channels = 3
    layers = []

    def flush():
        nonlocal layers
        modules.append(nn.Sequential(*layers))
        out_channels.append(in_channels)
        layers = []

    for v in cfg:
        stride = 1
        dilation = 1
        kernel_size = 3
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # Create new module with accumulated layers and flush layers list
            flush()
        elif v == 'S':
            conv_module = ConvModule(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, dilation=dilation,
                       activation=activation, bn=batch_norm)
            layers += nn.Sequential(*tuple([module for module in conv_module.children()]))
            # Create new module with accumulated layers and flush layers list
            flush()
        elif v == 'S3':
            conv_module = ConvModule(in_channels, in_channels, kernel_size=3, stride=3, padding=0, bias=False, dilation=dilation,
                       activation=activation, bn=batch_norm)
            layers += nn.Sequential(*tuple([module for module in conv_module.children()]))
            # Create new module with accumulated layers and flush layers list
            flush()
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            flush()
        else:
            if isinstance(v, tuple):
                assert len(v) == 3 or len(v) == 2, "Expected tuple in form depth, dilation, kernel_size"
                if len(v) == 3:
                    v, dilation, kernel_size = v
                else:
                    v, kernel_size = v

            padding = (kernel_size - 1) * dilation // 2
            conv_module = ConvModule(in_channels, v, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, dilation=dilation,
                       activation=activation, bn=batch_norm, channel_attention=channel_attention)
            mods = nn.Sequential(*tuple([module for module in conv_module.children()]))
            layers += mods
            in_channels = v

    assert len(layers) == 0

    return modules, out_channels


def make_resnet_modules(cfg, channel_attention=False):
    modules = nn.ModuleList()
    out_channels = []

    in_channels = 3
    layers = []

    def flush():
        nonlocal layers
        modules.append(nn.Sequential(*layers))
        out_channels.append(in_channels)
        layers = []

    in_layer_size = cfg[0]
    in_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_layer_size, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_layer_size),
            nn.ReLU()
        )
    layers.append(in_layer)
    in_channels = in_layer_size
    flush()
    layers.append(nn.Identity())
    flush()
    for v in cfg[1:]:
        v, downsample = v
        conv_module = ResBlock(in_channels, v, downsample=downsample, channel_attention=channel_attention)
        layers.append(conv_module)
        in_channels = v
        if downsample:
            flush()

    assert len(layers) == 0

    return modules, out_channels

def build_basketball_detector(model_name, scale, max_detections=100, player_threshold=0.0, nms_threshold=1.0, delta=3, c_r_kernel_size=3, fpn_depth=32, c_r_depth=32, channel_attention=False):
    # we have to pad feature map if using stride convs instead of poolings.
    padded_feature_map = False

    if model_name in basenet_configs.vgg_cfg:
        if "lrelu" in model_name:
            activation = nn.LeakyReLU
        elif "silu" in model_name:
            activation = nn.SiLU
        else:
            activation = nn.ReLU
        layers, out_channels = make_vgg_modules(basenet_configs.vgg_cfg[model_name], batch_norm=True, activation=activation, channel_attention="attention" in model_name)
    elif model_name in basenet_configs.efc_cfg:
        layers, out_channels = make_bm_modules(basenet_configs.efc_cfg[model_name], batch_norm=True)
    elif model_name in basenet_configs.resnet_cfg:
        layers, out_channels = make_resnet_modules(basenet_configs.resnet_cfg[model_name], channel_attention=channel_attention)
        padded_feature_map = True


    return_layer = int(math.log(scale, 2) - 1)
    base_net = FPN(layers, out_channels=out_channels, lateral_channels=fpn_depth,
                   return_layers=[return_layer])

    c_r_padding = (c_r_kernel_size - 1) // 2

    classifier = nn.Sequential(nn.Conv2d(fpn_depth, out_channels=c_r_depth, kernel_size=c_r_kernel_size, padding=c_r_padding),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(c_r_depth, out_channels=2, kernel_size=c_r_kernel_size, padding=c_r_padding))

    bbox_regressor = nn.Sequential(nn.Conv2d(fpn_depth, out_channels=c_r_depth, kernel_size=c_r_kernel_size, padding=c_r_padding),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(c_r_depth, out_channels=4, kernel_size=c_r_kernel_size, padding=c_r_padding))


    detector = BasketballDetector(model_name, base_net, bbox_regressor=bbox_regressor, classifier=classifier,
                                  player_threshold=player_threshold,
                                  max_player_detections=max_detections, player_downsampling=scale, player_delta=delta, nms_threshold=nms_threshold, padded_feature_map=padded_feature_map)

    return detector


if __name__ == "__main__":
    model = build_basketball_detector("res_8", c_r_kernel_size=3, delta=3, c_r_depth=128, fpn_depth=32, channel_attention=False)
    model.summary(with_arch=True)




