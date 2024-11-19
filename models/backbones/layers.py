import torch
import torch.nn as nn
from torch.nn import functional as F

class LK_encoder(nn.Module):
    def __init__(self, in_cs, out_cs, kernel_size=5, stride=1, padding=2):
        super(LK_encoder, self).__init__()
        self.in_cs = in_cs
        self.out_cs = out_cs
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.regular = nn.Sequential(
            nn.Conv2d(in_cs, out_cs, 3, 1, 1),
            nn.BatchNorm2d(out_cs),
        )
        self.large = nn.Sequential(
            nn.Conv2d(in_cs, out_cs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_cs),
        )
        self.one = nn.Sequential(
            nn.Conv2d(in_cs, out_cs, 1, 1, 0),
            nn.BatchNorm2d(out_cs),
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x1 = self.regular(x)
        x2 = self.large(x)
        x3 = self.one(x)
        if self.in_cs == self.out_cs and self.stride == 1:
            x = x1 + x2 + x3 + x
        else:
            x = x1 + x2 + x3
        return self.prelu(x)

class encoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=3, stride=1, padding=1):
        super(encoder, self).__init__()
        if kernel_size <= 3:
            self.layer = nn.Sequential(
                nn.Conv2d(in_cs, out_cs, kernel_size, stride, padding),
                nn.BatchNorm2d(out_cs),
                nn.PReLU()
            )
        elif kernel_size > 3:
            self.layer = LK_encoder(in_cs, out_cs, kernel_size, stride, padding)

    def forward(self, x):
        return self.layer(x)

class decoder(nn.Module):

    def __init__(self, in_cs, out_cs, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_cs, out_cs, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_cs),
            nn.PReLU()
        )

    def forward(self, x):
        return self.layer(x)


class affineOptimization3d(nn.Module):

    def __init__(self, alpha=1.):
        super(affineOptimization3d, self).__init__()

        self.alpha = alpha

    def get_matrix(self, x, y):

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        one = torch.ones(x.shape[0], 1, x.shape[2]).float().to(x.device)
        x = torch.cat([x, one], 1)

        out = torch.bmm(x, torch.transpose(x, -2, -1))
        inv = torch.inverse(out)
        out = torch.bmm(torch.transpose(x, -2, -1), inv)
        out = torch.bmm(y, out)

        return out

    def convert_xy(self, flow):
        dev = flow.device
        b,_,h,w,d = flow.shape
        y = torch.stack(torch.meshgrid([torch.arange(h,device=dev),torch.arange(w,device=dev),torch.arange(d,device=dev)]))
        y = y.unsqueeze(0).float().to(flow.device)
        x = y + flow
        x[:,0] = x[:,0] / (h-1) * 2 - 1
        x[:,1] = x[:,1] / (w-1) * 2 - 1
        x[:,2] = x[:,2] / (d-1) * 2 - 1
        y[:,0] = y[:,0] / (h-1) * 2 - 1
        y[:,1] = y[:,1] / (w-1) * 2 - 1
        y[:,2] = y[:,2] / (d-1) * 2 - 1
        y = y[:,[2,1,0]]
        x = x[:,[2,1,0]]
        x = x.view(b,3,-1).permute(0,2,1).contiguous()
        y = y.view(b,3,-1).permute(0,2,1).contiguous()

        return x, y

    def convert_pytorch_grid2flow(self, grid):

        grid = grid.permute(0,4,1,2,3)

        h, w, d = grid.shape[2:]
        grid_x = (grid[:,0:1, ...] + 1) * (d -1)/2
        grid_y = (grid[:,1:2, ...] + 1) * (w -1)/2
        grid_z = (grid[:,2:3, ...] + 1) * (h -1)/2

        grid = torch.cat([grid_z, grid_y, grid_x],dim=1)
        identity_grid = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w), torch.arange(d)])).to(grid.device).float().unsqueeze(0)
        flow = grid - identity_grid

        return flow

    def forward(self, flow, msk=None, alpha=None):

        if alpha == 0:
            return flow

        if alpha is None:
            alpha = self.alpha

        if msk is not None:
            msk = (msk==1).contiguous().view(1,-1)
            x, y = self.convert_xy(flow)

            x = x[msk].view(1,-1,3)
            y = y[msk].view(1,-1,3)
        else:
            x, y = self.convert_xy(flow)

        affine_ma = self.get_matrix(y, x)
        grid = F.affine_grid(affine_ma.float(), flow.size(), align_corners=False)
        affine_flow = self.convert_pytorch_grid2flow(grid) # (b,3,h,w,d), (b,3,h,w,d)

        flow = (1-alpha)*flow + alpha*affine_flow

        return flow

class affineOptimization2d(nn.Module):

    def __init__(self, alpha=1.):
        super(affineOptimization2d, self).__init__()

        self.alpha = alpha

    def get_matrix(self, x, y):

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        one = torch.ones(x.shape[0], 1, x.shape[2]).float().to(x.device)
        x = torch.cat([x, one], 1)

        out = torch.bmm(x, torch.transpose(x, -2, -1))
        inv = torch.inverse(out)
        out = torch.bmm(torch.transpose(x, -2, -1), inv)
        out = torch.bmm(y, out)

        return out

    def convert_xy(self, flow):
        b,_,h,w = flow.shape
        y = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w)]))
        y = y.unsqueeze(0).float().to(flow.device)
        x = y + flow
        x[:,0] = x[:,0] / (h-1) * 2 - 1
        x[:,1] = x[:,1] / (w-1) * 2 - 1
        y[:,0] = y[:,0] / (h-1) * 2 - 1
        y[:,1] = y[:,1] / (w-1) * 2 - 1
        y = y[:,[1,0]]
        x = x[:,[1,0]]
        x = x.view(b,2,-1).permute(0,2,1).contiguous()
        y = y.view(b,2,-1).permute(0,2,1).contiguous()

        return x, y

    def convert_pytorch_grid2flow(self, grid):

        grid = grid.permute(0,3,1,2)

        b,_, H, W = grid.shape
        grid_x = (grid[:,0:1, ...] + 1) * (W -1)/2
        grid_y = (grid[:,1:2, ...] + 1) * (H -1)/2

        grid = torch.cat([grid_y, grid_x],dim=1)
        identity_grid = torch.stack(torch.meshgrid([torch.arange(H), torch.arange(W)])).to(grid.device).float().unsqueeze(0)
        flow = grid - identity_grid

        return flow

    def forward(self, flow, msk=None, alpha=None):

        if alpha is None:
            alpha = self.alpha

        if msk is not None:
            msk = (msk>0).view(1,-1)
            x, y = self.convert_xy(flow)
            x = x[msk].view(1,-1,2)
            y = y[msk].view(1,-1,2)
        else:
            x, y = self.convert_xy(flow)

        affine_ma = self.get_matrix(y, x)
        # print(affine_ma)
        # quit()
        grid = F.affine_grid(affine_ma.float(), flow.size(), align_corners=False)
        affine_flow = self.convert_pytorch_grid2flow(grid) # (b,2,h,w), (b,2,h,w)

        flow = (1-self.alpha)*flow + self.alpha*affine_flow

        return flow