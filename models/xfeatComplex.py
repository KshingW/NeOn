import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BasicLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                                      nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                      nn.BatchNorm2d(out_channels, affine=False),
                                      nn.ReLU(inplace = True),
                                    )

    def forward(self, x):
      return self.layer(x)

class lk_layer(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=5, stride=1, padding=2):
        super(lk_layer, self).__init__()
        self.in_cs = in_cs
        self.out_cs = out_cs
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.regular = nn.Sequential(
            nn.Conv2d(in_cs, out_cs, 3, 1, 1),
            nn.BatchNorm2d(out_cs, affine=False),
        )
        self.large = nn.Sequential(
            nn.Conv2d(in_cs, out_cs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_cs, affine=False),
        )
        self.one = nn.Sequential(
            nn.Conv2d(in_cs, out_cs, 1, 1, 0),
            nn.BatchNorm2d(out_cs, affine=False),
        )
        self.prelu = nn.PReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.regular(x)
        x2 = self.large(x)
        x3 = self.one(x)
        if self.in_cs == self.out_cs and self.stride == 1:
            x = x1 + x2 + x3 + x
        else:
            x = x1 + x2 + x3
        return self.prelu(x)

def initialize_weights(m):
    """
    Initializes weights with a specific strategy, e.g., kaiming.
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)

def reinitialize_blocks(net):
    """
    Re-initialize specific modules in the network.
    """
    for name, module in net.named_modules():
        if 'block' in name:
            module.apply(initialize_weights)

class xfeatComplex(nn.Module):

    def __init__(self,
            pre_trained = '1',
            enable_grad = '0',
            zero_lk_branch = '0',
            lk_size = '7',
            fix_pretrained = '0',
        ):
        super().__init__()

        self.pre_trained = int(pre_trained)
        self.zero_lk_branch = int(zero_lk_branch)
        self.lk_size = int(lk_size)
        self.fix_pretrained = int(fix_pretrained)
        self.enable_grad = int(enable_grad)

        print('pre_trained: %d, zero_lk_branch: %d, lk_size: %d, fix_pretrained: %d, enable_grad: %d' % (self.pre_trained, self.zero_lk_branch, self.lk_size, self.fix_pretrained, self.enable_grad))

        self.norm = nn.InstanceNorm2d(1)

        self.skip1 = nn.Sequential(
            nn.AvgPool2d(4, stride = 4),
            nn.Conv2d (1, 24, 1, stride = 1, padding=0) 
        )

        self.block1 = nn.Sequential(
            BasicLayer( 1,  4, stride=1),
            BasicLayer( 4,  8, stride=2),
            BasicLayer( 8,  8, stride=1),
            BasicLayer( 8, 24, stride=2),
        )

        self.block2 = nn.Sequential(
            BasicLayer(24, 24, stride=1),
            BasicLayer(24, 24, stride=1),
        )

        self.block3 = nn.Sequential(
            BasicLayer(24, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, 1, padding=0),
        )
        self.block4 = nn.Sequential(
            BasicLayer(64, 64, stride=2),
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
        )

        self.block5 = nn.Sequential(
            BasicLayer( 64, 128, stride=2),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128, 128, stride=1),
            BasicLayer(128,  64, 1, padding=0),
        )

        self.block_fusion =  nn.Sequential(
            BasicLayer(64, 64, stride=1),
            BasicLayer(64, 64, stride=1),
            nn.Conv2d (64, 64, 1, padding=0)
        )

        self.heatmap_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d (64, 1, 1),
            nn.Sigmoid()
        )

        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            BasicLayer(64, 64, 1, padding=0),
            nn.Conv2d (64, 65, 1),
        )

        self.fine_matcher =  nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(inplace = True),
            nn.Linear(512, 64),
        )

        self.load_state_dict(torch.load('./xfeat.pt'))
        print('Loaded pretrained xfeat model.')

        if self.fix_pretrained == 1:
            # fix block layers
            for name, module in self.named_modules():
                if 'block' in name:
                    for param in module.parameters():
                        param.requires_grad = False

        if self.pre_trained == 0 and self.fix_pretrained == 0:
            reinitialize_blocks(self)

        ks = self.lk_size
        if self.zero_lk_branch == 1:
            self.lk_block3 = nn.Sequential(
                BasicLayer(24, 64, stride=2),
                lk_layer(64, 64, ks, 1, ks//2),
                lk_layer(64, 64, ks, 1, ks//2),
            )
            self.lk_block4 = nn.Sequential(
                BasicLayer(64, 64, stride=2),
                lk_layer(64, 64, ks, 1, ks//2),
                lk_layer(64, 64, ks, 1, ks//2),
            )
            self.lk_block5 = nn.Sequential(
                BasicLayer(64, 256, stride=2),
                lk_layer(256, 256, ks, 1, ks//2),
                lk_layer(256, 256, ks, 1, ks//2),
                BasicLayer(256, 64, 1, padding=0),
            )
            self.lk_block_fusion =  nn.Sequential(
                BasicLayer(64, 64, 1, padding=0),
                BasicLayer(64, 64, 1, padding=0),
                nn.Conv2d (64, 64, 1, padding=0)
            )

    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws).reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)

    def forward(self, y):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        #dont backprop through normalization
        if self.enable_grad == 0:
            with torch.no_grad():
                x = self.norm(y.mean(dim=1, keepdim = True))
        else:
            x = self.norm(y.mean(dim=1, keepdim = True))

        #main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        if self.zero_lk_branch == 1:
            lk_x3 = self.lk_block3(x2)
            lk_x4 = self.lk_block4(lk_x3)
            lk_x5 = self.lk_block5(lk_x4)

            lk_x4 = F.interpolate(lk_x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            lk_x5 = F.interpolate(lk_x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            lk_feats = self.lk_block_fusion( lk_x3 + lk_x4 + lk_x5 )

        #pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion( x3 + x4 + x5 )

        if self.zero_lk_branch == 1:
            feats = feats + lk_feats

        #heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

        return feats, keypoints, heatmap