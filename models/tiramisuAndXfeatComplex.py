import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.comir.tiramisu import DenseUNet
from models.xfeatComplex import xfeatComplex
from models.tiramisuComplex import tiramisuComplex

class tiramisuAndXfeatComplex(nn.Module):

    def __init__(self,
            ti_pretrained = '1',
            xf_pretrained = '1',
            enable_grad_xfeat = '0',
        ):
        super(tiramisuAndXfeatComplex, self).__init__()
        print('ti_pretrained: %s, xf_pretrained: %s' % (ti_pretrained, xf_pretrained))

        self.tiramisu = tiramisuComplex(out_cs='1', is_pretrained=ti_pretrained)
        self.xfeat = xfeatComplex(pre_trained=xf_pretrained, enable_grad=enable_grad_xfeat)

    def forward(self, x, y):

        pass