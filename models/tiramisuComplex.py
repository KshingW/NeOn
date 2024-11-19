import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbones.comir.tiramisu import DenseUNet

class tiramisuComplex(nn.Module):

    def __init__(self,
            out_cs = '1',
            is_pretrained = '0',
        ):
        super(tiramisuComplex, self).__init__()

        self.out_cs = int(out_cs)
        self.is_pretrained = int(is_pretrained)
        print('out_cs: %d, is_pretrained: %d' % (self.out_cs, self.is_pretrained))

        tiramisu_args = {
            "init_conv_filters": 32,
            "down_blocks": (4, 4, 4, 4, 4, 4),
            "up_blocks": (4, 4, 4, 4, 4, 4),
            "bottleneck_layers": 4,
            "upsampling_type": "upsample",
            "transition_pooling": "blurpool",
            "dropout_rate": 0.2,
            "early_transition": False,
            "activation_func": None,
            "compression": 0.75,
            "efficient": False,
            'include_top': False,
        }
        self.x_encoder = DenseUNet(in_channels=3, **tiramisu_args)
        self.y_encoder = DenseUNet(in_channels=1, **tiramisu_args)

        self.conv_x = nn.Conv2d(176,self.out_cs,1,1,0, bias=False)
        self.conv_y = nn.Conv2d(176,self.out_cs,1,1,0, bias=False)

        if self.is_pretrained and self.out_cs == 1:
            self.load_state_dict(torch.load('./tiramisu_cl.pth'))
            print('Pretrained model loaded from ./tiramisu_cl.pth')

    def forward(self, x, y):

        x_feas = self.x_encoder(x)
        y_feas = self.y_encoder(y)

        x_feas = self.conv_x(x_feas)
        y_feas = self.conv_y(y_feas)

        if self.out_cs > 1:
            x_feas = F.normalize(x_feas, p=2, dim=1)
            y_feas = F.normalize(y_feas, p=2, dim=1)

        return x_feas, y_feas