"""
Returns Unet3+ model
"""
from omegaconf import DictConfig

from .unet3plus import *
from .unet3plus_deep_supervision import unet3plus_deepsup
from .unet3plus_deep_supervision_cgm import unet3plus_deepsup_cgm

import sys
sys.path.append('/home/dragon1/seungeun2025/TransUNet-tf')

from transunet import TransUNet

from transunet.model import TransUNet_curvature, TransUNet_fused

def prepare_model(cfg: DictConfig, training=False, model_type=None, curvature=None):
    """
    Creates and return model object based on given model type.
    """

    input_shape = [cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.INPUT.CHANNELS]
    
    if cfg.MODEL.TYPE == "tiny_unet3plus":
        return tiny_unet3plus(
            input_shape,
            cfg.OUTPUT.CLASSES,
            training
        )
    elif cfg.MODEL.TYPE == "transunet":
        #  training parameter does not matter in this case
        if model_type == 'default':
            return TransUNet(image_size=224, pretrain=True)
            
        elif model_type == 'curv':
            return TransUNet_curvature(image_size=224, pretrain=True, curvature=curvature)
            
        elif model_type == 'fused':
            return TransUNet_fused(image_size=224, pretrain=True)
            
    elif cfg.MODEL.TYPE == "unet3plus_deepsup":
        return unet3plus_deepsup(
            input_shape,
            cfg.OUTPUT.CLASSES,
            training
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        if cfg.OUTPUT.CLASSES != 1:
            raise ValueError(
                "UNet3+ with Deep Supervision and Classification Guided Module"
                "\nOnly works when model output classes are equal to 1"
            )
        return unet3plus_deepsup_cgm(
            input_shape,
            cfg.OUTPUT.CLASSES,
            training
        )
    else:
        raise ValueError(
            "Wrong model type passed."
            "\nPlease check config file for possible options."
        )
