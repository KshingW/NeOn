import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.functions import SpatialTransformer
from models.backbones.layers import encoder, affineOptimization2d

class dispWarp(nn.Module):

    def __init__(self, in_cs=1, N_s=32, out_cs=2, lk_size=3):

        super(dispWarp, self).__init__()

        self.disp_field = nn.Sequential(
            encoder(2*in_cs, N_s, lk_size, 1, lk_size//2),
            encoder(N_s, 2*N_s, lk_size, 1, lk_size//2),
        )
        self.flow = nn.Conv2d(2*N_s, out_cs, 3, 1, 1)
        self.init_zero_flow()

    def init_zero_flow(self):
        self.flow.weight.data.fill_(0)
        self.flow.bias.data.fill_(0)

    def forward(self, x, y):

        flow = self.disp_field(torch.cat((y+x,y-x), dim=1))
        flow = self.flow(flow)

        return flow

class smallDispModelComplex(nn.Module):

    def __init__(self, 
        N_s = '32', 
        lk_size = '5', 
        img_size = '(832,832)'
    ):
        super(smallDispModelComplex, self).__init__()

        self.N_s = int(N_s)
        self.lk_size = int(lk_size)
        self.ss = eval(img_size)
        N_s = self.N_s
        lk_size = self.lk_size
        ss = self.ss

        print("N_s: %d, lk_size: %d, img_size: %s" % (self.N_s, self.lk_size, self.ss))

        self.disp_field_0 = dispWarp(in_cs=N_s, N_s=N_s*1, out_cs=2, lk_size=lk_size)
        self.disp_field_1 = dispWarp(in_cs=N_s, N_s=N_s*2, out_cs=2, lk_size=lk_size)
        self.disp_field_2 = dispWarp(in_cs=N_s, N_s=N_s*4, out_cs=2, lk_size=lk_size)
        self.disp_field_3 = dispWarp(in_cs=N_s, N_s=N_s*8, out_cs=2, lk_size=lk_size)
        self.disp_field_4 = dispWarp(in_cs=N_s, N_s=N_s*8, out_cs=2, lk_size=lk_size)

        self.transformer_4 = SpatialTransformer([s // 16 for s in ss])
        self.transformer_3 = SpatialTransformer([s // 8 for s in ss])
        self.transformer_2 = SpatialTransformer([s // 4 for s in ss])
        self.transformer_1 = SpatialTransformer([s // 2 for s in ss])
        self.transformer_0 = SpatialTransformer([s // 1 for s in ss])

        self.aff_opt = affineOptimization2d(alpha=1.)

        self.pre_conv = nn.Sequential(
            encoder(1, N_s, 3, 1, 1),
            encoder(N_s, 2*N_s, 3, 1, 1),
            encoder(2*N_s, N_s, 3, 1, 1),
        )

    def forward(self, x, y, y_msk):

        x = self.pre_conv(x)
        y = self.pre_conv(y)

        x0, y0, y_msk0 = x, y, y_msk
        x1 = F.interpolate(x0, scale_factor=0.5, mode='bilinear', align_corners=True)
        y1 = F.interpolate(y0, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk1 = F.interpolate(y_msk0, scale_factor=0.5, mode='nearest')
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)
        y2 = F.interpolate(y1, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk2 = F.interpolate(y_msk1, scale_factor=0.5, mode='nearest')
        x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=True)
        y3 = F.interpolate(y2, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk3 = F.interpolate(y_msk2, scale_factor=0.5, mode='nearest')
        x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=True)
        y4 = F.interpolate(y3, scale_factor=0.5, mode='bilinear', align_corners=True)
        y_msk4 = F.interpolate(y_msk3, scale_factor=0.5, mode='nearest')

        pflow_4 = self.disp_field_2(x4, y4)
        flow_4 = self.aff_opt(pflow_4, y_msk4)
        up_flow_4 = F.interpolate(flow_4, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x3 = self.transformer_3(x3, up_flow_4)

        pflow_3 = self.disp_field_2(warped_x3, y3)
        flow_3 = pflow_3 + self.transformer_3(up_flow_4, pflow_3)
        flow_3 = self.aff_opt(flow_3, y_msk3)
        up_flow_3 = F.interpolate(flow_3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x2 = self.transformer_2(x2, up_flow_3)

        pflow_2 = self.disp_field_2(warped_x2, y2)
        flow_2 = pflow_2 + self.transformer_2(up_flow_3, pflow_2)
        flow_2 = self.aff_opt(flow_2, y_msk2)
        up_flow_2 = F.interpolate(flow_2, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x1 = self.transformer_1(x1, up_flow_2)

        pflow_1 = self.disp_field_2(warped_x1, y1)
        flow_1 = pflow_1 + self.transformer_1(up_flow_2, pflow_1)
        flow_1 = self.aff_opt(flow_1, y_msk1)
        up_flow_1 = F.interpolate(flow_1, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x0 = self.transformer_0(x0, up_flow_1)

        pflow_0 = self.disp_field_2(warped_x0, y0)
        flow_0 = pflow_0 + self.transformer_0(up_flow_1, pflow_0)
        flow_0 = self.aff_opt(flow_0, y_msk0)

        pflow = [pflow_0, pflow_1, pflow_2, pflow_3, pflow_4]
        flows = [flow_0, flow_1, flow_2, flow_3, flow_4]

        xs = [x0, x1, x2, x3, x4]
        ys = [y0, y1, y2, y3, y4]

        return flows, xs, ys