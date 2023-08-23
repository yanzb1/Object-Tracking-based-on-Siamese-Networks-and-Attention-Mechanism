# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.models.neck.neck import AdjustLayer, AdjustAllLayer
from siamban.models.neck.featurefusion_network import FeatureFusionNetwork
NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer
        }

NETWORKS={
    'featurefusion_network':FeatureFusionNetwork
}
def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
def get_fearure(name, **kwargs):
    return NETWORKS[name](**kwargs)