import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from PIL import Image

from utils.functions import registerSTModel, compute_tre, SpatialTransformer, GaussianBlur2D, get_downsampled_images,AverageMeter
from models.backbones.layers import affineOptimization2d, encoder
from utils.loss import NccLoss, Grad2d
import time

def read_img_kps(opt):

    kps = pd.read_csv(opt['kps_fp'], header=None, delim_whitespace=True)
    
    img_x = np.array(Image.open(opt['x_fp']))[...,1]
    img_y = np.array(Image.open(opt['y_fp']))[...,1]
    img_y = 255-img_y
    img_y[img_y==255] = 0 
    print(opt['y_fp'])
    print(img_x.shape, img_y.shape)
    kps_x = kps[[2, 3]].values
    kps_y = kps[[0, 1]].values

    img_x = torch.tensor(img_x).unsqueeze(0).unsqueeze(0).float() / 255
    img_y = torch.tensor(img_y).unsqueeze(0).unsqueeze(0).float() / 255

    return img_x, img_y, kps_x, kps_y

class dispWarp(nn.Module):

    def __init__(self, in_cs=1, N_s=32, out_cs=2, lk_size=3):

        super(dispWarp, self).__init__()

        self.disp_field = nn.Sequential(
            encoder(2*in_cs, 2*N_s, lk_size, 1, lk_size//2),
            encoder(2*N_s, 2*N_s, lk_size, 1, lk_size//2),
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

class smallDispModel(nn.Module):

    def __init__(self, N_s=32, lk_size=3, ss=(832,832)):

        super(smallDispModel, self).__init__()

        self.disp_field = dispWarp(in_cs=N_s, N_s=N_s*4, out_cs=2, lk_size=lk_size)

        self.transformer_4 = SpatialTransformer([s // 16 for s in ss])
        self.transformer_3 = SpatialTransformer([s // 8 for s in ss])
        self.transformer_2 = SpatialTransformer([s // 4 for s in ss])
        self.transformer_1 = SpatialTransformer([s // 2 for s in ss])
        self.transformer_0 = SpatialTransformer([s // 1 for s in ss])

        self.aff_opt = affineOptimization2d(alpha=0.3)

        self.pre_conv1 = nn.Sequential(
            encoder(1, N_s, 3, 1, 1),
            encoder(N_s, 2*N_s, 3, 1, 1),
            encoder(2*N_s, N_s, 3, 1, 1),
        )
        self.pre_conv2 = nn.Sequential(
            encoder(1, N_s, 3, 1, 1),
            encoder(N_s, 2*N_s, 3, 1, 1),
            encoder(2*N_s, N_s, 3, 1, 1),
        )

        self.blur = GaussianBlur2D(1, 1).cuda()

    def forward(self, x, y, y_msk):

        x = self.pre_conv1(x)
        y = self.pre_conv2(y)

        x0, y0, y_msk0 = x, y, y_msk
        x1 = F.interpolate(x0, scale_factor=0.5, mode='bilinear', align_corners=True)
        y1 = F.interpolate(y0, scale_factor=0.5, mode='bilinear', align_corners=True)
        bluredd_y_msk0 = self.blur(y_msk0)
        y_msk1 = F.interpolate(bluredd_y_msk0, scale_factor=0.5, mode='nearest')
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)
        y2 = F.interpolate(y1, scale_factor=0.5, mode='bilinear', align_corners=True)
        bluredd_y_msk1 = self.blur(y_msk1)
        y_msk2 = F.interpolate(bluredd_y_msk1, scale_factor=0.5, mode='nearest')

        pflow_2 = self.disp_field(x2, y2)
        flow_2 = self.aff_opt(pflow_2, y_msk2)
        up_flow_2 = F.interpolate(flow_2, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x1 = self.transformer_1(x1, up_flow_2)

        pflow_1 = self.disp_field(warped_x1, y1)
        flow_1 = pflow_1 + up_flow_2
        flow_1 = self.aff_opt(flow_1, y_msk1)
        up_flow_1 = F.interpolate(flow_1, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped_x0 = self.transformer_0(x0, up_flow_1)

        pflow_0 = self.disp_field(warped_x0, y0)
        flow_0 = pflow_0 + up_flow_1
        flow_0 = self.aff_opt(flow_0, y_msk0)

        pflow = [pflow_0, pflow_1, pflow_2]
        flows = [flow_0, flow_1, flow_2]

        xs = [x0, x1, x2]
        ys = [y0, y1, y2]

        return pflow, flows, xs, ys

def run(opt):

    fea_x = torch.load(opt['fea_x_fp']).cuda().unsqueeze(0)
    fea_y = torch.load(opt['fea_y_fp']).cuda().unsqueeze(0)
    
    fea_x = F.interpolate(fea_x, size=opt['img_size'], mode='bilinear', align_corners=True)
    fea_y = F.interpolate(fea_y, size=opt['img_size'], mode='bilinear', align_corners=True)

    seg_x = torch.load(opt['seg_x_fp']).cuda().unsqueeze(0)
    seg_y = torch.load(opt['seg_y_fp']).cuda().unsqueeze(0)

    # seg_x = torch.argmax(seg_x, dim=1, keepdim=True).float()
    # seg_y = torch.argmax(seg_y, dim=1, keepdim=True).float()
    # msk = (seg_y == 1).float()  # get the vessel mask
    seg_y = torch.sigmoid(seg_y).float()
    msk = (seg_y>0.99).float()

    print("Segmentation Features Loaded")

    img_x, img_y, kps_x, kps_y = read_img_kps(opt)    
    reg_model_bi = registerSTModel(opt['ori_size'], 'bilinear').cuda()
    img_x, img_y = img_x.cuda(), img_y.cuda()
    img_x=F.interpolate(img_x, size=opt['img_size'], mode='bilinear', align_corners=True)
    img_y=F.interpolate(img_y, size=opt['img_size'], mode='bilinear', align_corners=True)
    disk_msk = (img_y > 1e-4)
    msk_y = torch.tensor(np.array(Image.open(opt['msk_y']))[...,1]).unsqueeze(0).unsqueeze(0).cuda()
    img_x = img_x * msk_y
    img_y = img_y * msk_y

    print("Images and Keypoints Loaded")

    flow_model = smallDispModel(N_s=32, lk_size=11, ss=opt['img_size']).cuda()
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)
    flow_model.train()
    print('flow model initialized')

    creterion_ncc_0 = NccLoss(win=[19,19])
    creterion_ncc_1 = NccLoss(win=[13,13])
    creterion_ncc_2 = NccLoss(win=[9,9])
    creterion_mse = nn.MSELoss()
    creterion_reg = Grad2d()


    best_tre=1000
    best_tre_iter=0
    best_ncc_iter=0
    best_ncc = -1
    time_s = time.time()
    niter = 300 if opt['x_fp'].split('/')[-1].split('-')[0] != 'normal_1' else 300
    print(niter)
    for idx in range(niter):
        pflows, flows, xs, ys = flow_model(img_x, img_y, msk)
        xs = get_downsampled_images(img_x, 4, mode='bilinear')
        ys = get_downsampled_images(img_y, 4, mode='bilinear')
        msks = get_downsampled_images(disk_msk.float(), 4, mode='bilinear')

        sim_loss_0 = creterion_ncc_0(flow_model.transformer_0(xs[0], flows[0])*msks[0], ys[0]*msks[0])
        sim_loss_1 = creterion_ncc_1(flow_model.transformer_1(xs[1], flows[1])*msks[1], ys[1]*msks[1]) / 2
        sim_loss_2 = creterion_ncc_2(flow_model.transformer_2(xs[2], flows[2])*msks[2], ys[2]*msks[2]) / 4
        sim_loss = sim_loss_0 + sim_loss_1 + sim_loss_2

        # sim_loss_0 = creterion_mse(flow_model.transformer_0(xs[0], flows[0])*msks[0], ys[0]*msks[0])
        # sim_loss_1 = creterion_mse(flow_model.transformer_1(xs[1], flows[1])*msks[1], ys[1]*msks[1]) / 2
        # sim_loss_2 = creterion_mse(flow_model.transformer_2(xs[2], flows[2])*msks[2], ys[2]*msks[2]) / 4
        # sim_loss = sim_loss_0 + sim_loss_1 + sim_loss_2

        reg_loss_0 = creterion_reg(pflows[0])
        reg_loss_1 = creterion_reg(pflows[1]) / 2
        reg_loss_2 = creterion_reg(pflows[2]) / 4
        reg_loss = reg_loss_0 + reg_loss_1 + reg_loss_2

        loss = opt['sim_w']*sim_loss + opt['reg_w']*reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_flow = flows[0]
        pos_flow = pred_flow
        pos_flow = F.interpolate(pos_flow, size=opt['ori_size'], mode='bilinear', align_corners=True)
        pos_flow = pos_flow * opt['ori_size'][0] / opt['img_size'][0]

        test_x = F.interpolate(img_x, size=opt['ori_size'], mode='bilinear', align_corners=True)
        test_y = F.interpolate(img_y, size=opt['ori_size'], mode='bilinear', align_corners=True)
        warped_x = reg_model_bi(test_x.float(), pos_flow)  # 1,1,h,w
        test_y[warped_x==0]=0
        warped_x[test_y==0]=0
        ncc_img_criterion = NccLoss()
        ncc_img_loss = ncc_img_criterion(test_y, warped_x)
        ncc_score = 1 - ncc_img_loss


        pos_flow = pos_flow.detach().cpu().numpy().squeeze()
        px = kps_x.astype(int)
        py = kps_y.astype(int)
        dx = pos_flow[1,py[:,1], py[:,0]]
        dy = pos_flow[0,py[:,1], py[:,0]]
        disp = np.array([dx,dy]).transpose()
        warped_kps = kps_y + disp
        eval_tre = compute_tre(warped_kps, kps_x, opt['spacing']).mean()
        init_tre = compute_tre(kps_x, kps_y, opt['spacing']).mean()
        eval_tre = eval_tre
        init_tre = init_tre

        if eval_tre < best_tre:
            best_tre = eval_tre
            best_tre_iter = idx
            sub_id = opt['x_fp'].split('/')[-1].split('_')[0]
            id_ = opt['x_fp'].split('/')[-1].split('.')[0].split('_')[1]
            np.save(f'{sub_id}_{id_}_best_field_tre.npy', pos_flow)
        if ncc_score > best_ncc:
            best_ncc = ncc_score
            best_ncc_iter = idx
            sub_id = opt['x_fp'].split('/')[-1].split('_')[0]
            id_ = opt['x_fp'].split('/')[-1].split('.')[0].split('_')[1]
            np.save(f'{sub_id}_{id_}_best_field.npy', pos_flow)
        print('iter: %d, init_tre: %.4f, eval_tre: %.4f, best_tre: %.4f, eval_ncc: %.4f, best_ncc: %.4f,  best_iter_tre: %d, best_iter_ncc: %d' % (idx, init_tre, eval_tre, best_tre, ncc_score, best_ncc, best_tre_iter, best_ncc_iter))
    time_e = time.time() - time_s
    return init_tre, best_tre, best_tre_iter, best_ncc_iter, time_e

if __name__ == '__main__':

    opt = {
        'spacing': (1.,1.),
        'fea_x_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/A01_1_feature.pt',
        'fea_y_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/A01_2_feature.pt',
        'seg_x_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/A01_1_logit.pt',
        'seg_y_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/A01_2_logit.pt',
        'x_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/A01_1.jpg',
        'y_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/A01_2.jpg',
        'kps_fp': '/home/jcw/Documents/GitHub/aaai25/FIRE_sample/control_points_A01_1_2.txt',
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("--ori_size", type = str, default = '(576,720)')
    parser.add_argument("--img_size", type = str, default = '(576,720)')
    parser.add_argument("--niter", type = int, default = 25)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--sim_w", type = float, default = 1)
    parser.add_argument("--reg_w", type = float, default = 1)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = str(opt['img_size'])
    opt['img_size'] = eval(opt['img_size'])
    opt['ori_size'] = eval(opt['ori_size'])

    # run(opt)

    pre_align_CF = '/scratch/dl3837/neuralforce/neon_oct/neon_cf-fa/prealign_cnn'
    pre_align_img = '/scratch/dl3837/neuralforce/neon_oct/neon_cf-fa/prealign_img'
    pre_align_txt = '/scratch/dl3837/neuralforce/neon_oct/neon_cf-fa/prealign_gt'

    normal_list = [f"normal_{i}" for i in range(1, 29)]
    abnormal_list = [f"abnormal_{i}" for i in range(1, 31)]

    # Combine both lists
    sub_list = normal_list + abnormal_list

    final_list = []
    sub_list = ['normal_1','normal_12']
    for sub in sub_list:
        print(sub)
        if sub == 'normal_1':
            opt['niter'] = 300
        sub_id = sub.split('_')[1]
        modal = sub.split('_')[0]
        opt.update({
            'fea_y_fp': f'{pre_align_CF}/{sub}_feature.pt',
            'fea_x_fp': f'{pre_align_CF}/{sub}-{sub_id}_feature.pt',
            'seg_y_fp': f'{pre_align_CF}/{sub}_logit.pt',
            'seg_x_fp': f'{pre_align_CF}/{sub}-{sub_id}_logit.pt',
            'y_fp': f'{pre_align_img}/{sub}.jpg',
            'x_fp': f'{pre_align_img}/{sub}-{sub_id}.jpg',
            'kps_fp': f'{pre_align_txt}/{sub}.txt',
            'msk_y': f'{pre_align_img}/{modal}_mask_{sub_id}.jpg',
        })
        init_tre, best_tre, best_iter, best_iter_ncc, time_e = run(opt)
        final_list.append([init_tre, best_tre, best_iter, best_iter_ncc, time_e])

    
'''
python test_oct_neon.py --ori_size '(2912,2912)' --img_size '(1024,1024)' temp=0.001 ks=1 alpha=1

*=* is the argument for model
'''
