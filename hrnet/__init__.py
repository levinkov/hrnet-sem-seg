# ------------------------------------------------------------------------------
# Copyright (c) Evgeny Levinkov (evgeny.levinkov@gmail.com)
# Licensed under the MIT License.
# ------------------------------------------------------------------------------


import os
from .seg_hrnet import HighResolutionNet
from yacs.config import CfgNode as CN


def get_hrnet(config_name, num_classes, num_input_channels=3):
    '''
    Returns an instance of HRnet.

    :param config_name: can one of the following: small_v1, small_v2, big.
    :param num_classes: number of predicted semantic classes.
    :param num_input_channels: number of input channels.
    '''

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.EXTRA = CN(new_allowed=True)

    cfg.defrost()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

    if config_name == 'small_v1':
        cfg.merge_from_file(os.path.join(path, 'seg_hrnet_w18_small_v1.yaml'))
    elif config_name == 'small_v2':
        cfg.merge_from_file(os.path.join(path, 'seg_hrnet_w18_small_v2.yaml'))
    elif config_name == 'big':
        cfg.merge_from_file(os.path.join(path, 'seg_hrnet_w48.yaml'))
    else:
        raise RuntimeError(f'HRNet: config name \'{config_name}\' is not currently supported.')

    cfg.freeze()

    model = HighResolutionNet(cfg, num_classes, num_input_channels)
    model.init_weights()

    return model
