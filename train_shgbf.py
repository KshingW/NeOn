import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from utils.loss import batched_NCE_half_loss, random_patches_msk
from utils import getters, setters
from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, normalize_common_torch, AffineKeypointAligner, find_matches
from utils.affine_agumentation import random_affine_augment

from models.xfeat import XFeat
from models.xfeatComplex import xfeatComplex
from utils.xfeat_augmentation import AugmentationPipe
from utils.xfeat_utils import get_corresponding_pts, check_accuracy
from utils.xfeats_losses import dual_softmax_loss, coordinate_classification_loss, keypoint_loss

def get_flow(xfeat_wrap, img_x, img_y):

    dev = img_x.device
    h,w = img_x.shape[2:]

    kps_x, kps_y = xfeat_wrap.match_xfeat(img_x, img_y, top_k = 16000)
    kps_x, kps_y  = find_matches(kps_x, kps_y)

    kps_x_ = torch.from_numpy(kps_x).unsqueeze(0).float().to(dev)
    kps_y_ = torch.from_numpy(kps_y).unsqueeze(0).float().to(dev)

    kps_x_ = kps_x_ / (h-1) * 2 - 1
    kps_y_ = kps_y_ / (w-1) * 2 - 1

    affine_model = AffineKeypointAligner(2)
    pos_flow = affine_model.flow_from_points(kps_x_, kps_y_, (1,2,h,w))

    return pos_flow

def simple_bilinear_interpolation(b, feats, pts):

    pts_l_0 = pts[:,1].floor().long()
    pts_r_0 = pts[:,0].floor().long()
    pts_l_1 = pts_l_0 + 1
    pts_r_1 = pts_r_0 + 1

    c00 = (pts_l_1 - pts[:,1]) * (pts_r_1 - pts[:,0])
    c01 = (pts_l_1 - pts[:,1]) * (pts[:,0] - pts_r_0)
    c10 = (pts[:,1] - pts_l_0) * (pts_r_1 - pts[:,0])
    c11 = (pts[:,1] - pts_l_0) * (pts[:,0] - pts_r_0)

    m00 = feats[b, :, pts_l_0, pts_r_0].permute(1,0) * c00.unsqueeze(1)
    m01 = feats[b, :, pts_l_0, pts_r_1].permute(1,0) * c01.unsqueeze(1)
    m10 = feats[b, :, pts_l_1, pts_r_0].permute(1,0) * c10.unsqueeze(1)
    m11 = feats[b, :, pts_l_1, pts_r_1].permute(1,0) * c11.unsqueeze(1)

    return m00 + m01 + m10 + m11

def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    train_loader = getters.getDataLoader(opt, split='train')
    val_loader = getters.getDataLoader(opt, split='val')
    model, init_epoch = getters.getTrainModelWithCheckpoints(opt)
    model_saver = getters.getModelSaver(opt)

    reg_model_bi = registerSTModel(opt['img_size'], 'bilinear').cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)

    xfeat_wrap = XFeat(device = opt['device'])

    augmentor = AugmentationPipe(
        device = opt['device'],
        batch_size = opt['batch_size'],
        out_resolution = opt['img_size'], 
        warp_resolution = opt['img_size'],
        sides_crop = 0.1,
        photometric = True,
        geometric = True,
    )

    best_tre = 1e5
    best_epoch = 0
    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        loss_all = AverageMeter()
        loss_nce_x2y_all = AverageMeter()
        loss_nce_y2x_all = AverageMeter()
        loss_ds_all = AverageMeter()
        loss_coords_all = AverageMeter()
        loss_kp_all = AverageMeter()
        eval_acc_coarse_all = AverageMeter()
        eval_acc_coords_all = AverageMeter()

        for idx, data in enumerate(train_loader):
            model.train()
            data = [Variable(t.cuda()) for t in data[:4]]
            x, y = data[0].float(), data[1].float()
            x_msk, y_msk = data[2].float(), data[3].float()
            aug_params = opt['affine_aug_params']
            aug_params = [aug_params['scale'], aug_params['offset'], aug_params['theta'], aug_params['shear']]

            x, y, x_msk, y_msk = random_affine_augment(x, y, seg=x_msk, seg2=y_msk, max_random_params=aug_params)
            x, y = x[...,:832,:832], y[...,:832,:832] # crop from 834 to 832
            x_msk, y_msk = x_msk[...,:832,:832], y_msk[...,:832,:832] # crop from 834 to 832

            x_feas, y_feas = model.tiramisu(x, y)

            patch_s = opt['patch_size']
            if opt['is_msk']:
                x1, y1 = random_patches_msk(x_feas, y_feas, opt['n_samples'], patch_s, x_msk) # (b,n1/b,1,h,w)
                x2, y2 = random_patches_msk(x_feas, y_feas, opt['n_samples'], patch_s, y_msk) # (b,n2/b,1,h,w)
            else:
                x1, y1 = random_patches_msk(x_feas, y_feas, opt['n_samples'], patch_s, None) # (b,n1/b,1,h,w)
                x2, y2 = random_patches_msk(x_feas, y_feas, opt['n_samples'], patch_s, None) # (b,n2/b,1,h,w)
            x1 = x1.contiguous().view(-1,1,patch_s,patch_s) # (n1, 1, h, w)
            y1 = y1.contiguous().view(-1,1,patch_s,patch_s) # (n2, 1, h, w)
            x2 = x2.contiguous().view(-1,1,patch_s,patch_s) # (n1, 1, h, w)
            y2 = y2.contiguous().view(-1,1,patch_s,patch_s) # (n2, 1, h, w)
            n1, n2 = len(x1), len(x2)

            x1 = F.avg_pool2d(x1, kernel_size=5, stride=2, padding=2, count_include_pad=False)
            y1 = F.avg_pool2d(y1, kernel_size=5, stride=2, padding=2, count_include_pad=False)
            x2 = F.avg_pool2d(x2, kernel_size=5, stride=2, padding=2, count_include_pad=False)
            y2 = F.avg_pool2d(y2, kernel_size=5, stride=2, padding=2, count_include_pad=False)
            x1, x2 = x1.view(1,n1,-1), x2.view(1,n2,-1)
            y1, y2 = y1.view(1,n1,-1), y2.view(1,n2,-1)

            if opt['is_stacked_pixel_nce'] == 0:
                x1, x2 = x1.contiguous().permute(0,2,1), x2.contiguous().permute(0,2,1)
                y1, y2 = y1.contiguous().permute(0,2,1), y2.contiguous().permute(0,2,1)

            loss_nce_x2y = batched_NCE_half_loss(x1, y1, tau=opt['tau'], crtic=opt['critic']).mean() * opt['nce_w']
            loss_nce_y2x = batched_NCE_half_loss(y2, x2, tau=opt['tau'], crtic=opt['critic']).mean() * opt['nce_w']
            loss_nce_x2y_all.update(loss_nce_x2y.item(), x_feas.numel())
            # loss_nce_y2x_all.update(loss_nce_y2x.item(), x_feas.numel())

            x_feas, y_feas = normalize_common_torch(x_feas, y_feas)
            x_feas, y_feas = x_feas.repeat(1,3,1,1), y_feas.repeat(1,3,1,1)

            p1, H1 = augmentor(x_feas, difficulty=0.1)
            p2, H2 = augmentor(y_feas, difficulty=0.1, TPS = True, prob_deformation = 0.7)

            h_coarse, w_coarse = p1[0].shape[-2] // 8, p1[0].shape[-1] // 8
            _ , positives_c = get_corresponding_pts(p1,p2,H1,H2,augmentor,h_coarse,w_coarse)

            feats1, kpts1, hmap1 = model.xfeat(p1)
            feats2, kpts2, hmap2 = model.xfeat(p2)

            loss_ds = 0
            loss_coords = 0
            loss_kp = 0
            acc_coarse = 0
            acc_coords = 0
            for b in range(len(positives_c)):
                #Get positive correspondencies
                pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                #Grab features at corresponding idxs
                if opt['is_bi']:
                    m1 = simple_bilinear_interpolation(b, feats1, pts1)
                    m2 = simple_bilinear_interpolation(b, feats2, pts2)
                else:
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)
                #grab heatmaps at corresponding idxs
                if opt['is_bi']:
                    h1 = simple_bilinear_interpolation(b, hmap1, pts1)
                    h2 = simple_bilinear_interpolation(b, hmap2, pts2)
                else:
                    h1 = hmap1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    h2 = hmap2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)
                coords1 = model.xfeat.fine_matcher(torch.cat([m1, m2], dim=-1))

                #Compute losses
                loss_ds_, conf = dual_softmax_loss(m1, m2)
                loss_coords_, acc_coords_ = coordinate_classification_loss(coords1, pts1, pts2, conf)
                loss_kp_ =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                loss_ds = loss_ds + loss_ds_/len(positives_c)
                loss_coords = loss_coords + loss_coords_/len(positives_c)
                loss_kp = loss_kp + loss_kp_/len(positives_c)

                loss_ds_all.update(loss_ds_.item(), len(positives_c))
                loss_coords_all.update(loss_coords_.item(), len(positives_c))
                loss_kp_all.update(loss_kp_.item(), len(positives_c))

                acc_coarse_ = check_accuracy(m1, m2)
                acc_coarse = acc_coarse + acc_coarse_/len(positives_c)
                acc_coords = acc_coords + acc_coords_/len(positives_c)
                eval_acc_coarse_all.update(acc_coarse_, len(positives_c))
                eval_acc_coords_all.update(acc_coords_, len(positives_c))

            loss = loss_nce_x2y + loss_nce_y2x + loss_ds + loss_coords + loss_kp

            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, loss_x2y {:.4f}, loss_y2x {:.4f}, loss_ds {:.4f}, loss_coords {:.4f}, loss_kp {:.4f}, acc_coa {:.4f}, acc_cds {:.4f}'.format(idx+1, len(train_loader), loss.item(), loss_nce_x2y.item(), loss_nce_y2x.item(), loss_ds.item(), loss_coords.item(), loss_kp.item(), acc_coarse, acc_coords), end='\r', flush=True)

        print('---->>>> Epoch {} train loss {:.4f}, loss_nce_x2y {:.4f}, loss_nce_y2x {:.4f}, loss_ds {:.4f}, loss_coords {:.4f}, loss_kp {:.4f}, acc_coa {:.4f}, acc_cds {:.4f}'.format(epoch+1, loss_all.avg, loss_nce_x2y_all.avg, loss_nce_y2x_all.avg, loss_ds_all.avg, loss_coords_all.avg, loss_kp_all.avg, eval_acc_coarse_all.avg, eval_acc_coords_all.avg), flush=True)

        '''
        Validation
        '''
        eval_tre = AverageMeter()
        init_tre = AverageMeter()
        with torch.no_grad():
            xfeat_wrap.init_net(model.xfeat)
            for data in val_loader:
                model.eval()
                data = [Variable(t.cuda()) for t in data[:6]]
                x_imgs, y_imgs = data[0].float(), data[1].float()
                x_coords, y_coords = data[2], data[3]
                x_imgs, y_imgs = x_imgs[...,0:832,0:832], y_imgs[...,0:832,0:832] # crop from 834 to 832

                x_imgs, y_imgs = model.tiramisu(x_imgs, y_imgs)
                x_imgs, y_imgs = normalize_common_torch(x_imgs, y_imgs)
                pred_flow = get_flow(xfeat_wrap, x_imgs, y_imgs)

                x_coords = torch.clip(x_coords, 0, 831)
                y_coords = torch.clip(y_coords, 0, 831)
                kps_x_ = x_coords.squeeze()
                kps_y_ = y_coords.squeeze()
                py = kps_y_.long()

                dx = pred_flow[0,1][py[:,1], py[:,0]]
                dy = pred_flow[0,0][py[:,1], py[:,0]]

                disp = torch.stack([dx, dy], dim=1)
                warped_y = kps_y_ + disp

                init_error = torch.mean(torch.norm(kps_y_ - kps_x_, dim=-1))
                eval_error = torch.mean(torch.norm(warped_y - kps_x_, dim=-1))
                print('Initial error: %.4f, Eval error: %.4f' % (init_error, eval_error), end='\r', flush=True)

                init_tre.update(init_error.item(), kps_x_.numel())
                eval_tre.update(eval_error.item(), kps_x_.numel())

        if eval_tre.avg < best_tre:
            best_tre = eval_tre.avg
            best_epoch = epoch

        print('Epoch {} init tre {:.4f}, eval tre {:.4f}, best tre {:.4f} at epoch {}'.format(epoch+1, init_tre.avg, eval_tre.avg, best_tre, best_epoch), flush=True)

        model_saver.saveModel(model, epoch, -eval_tre.avg)
        loss_all.reset()

if __name__ == '__main__':

    opt = {
        'logs_path': './logs',       # path to save logs
        'save_freq': 2,              # save model every save_freq epochs
        'n_checkpoints': 6,          # number of checkpoints to keep
        'power': 0.9,                # decay power
        'num_workers': 0,            # number of workers for data loading
        'affine_aug_params': {
            'scale': 0.15,
            'offset': 0.15,
            'theta': 3.1416/4,
            'shear': 0.15,
        }
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'tiramisuAndXfeatComplex111Msk1Ps128')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'ShgBfReg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "/home/jackywang/Documents/Datasets/")
    parser.add_argument("--epochs", type = int, default = 201)
    parser.add_argument("--nce_w", type = float, default = 0.1)
    parser.add_argument("--lr", type = float, default = 4e-4)
    parser.add_argument("--img_size", type = str, default = '(832,832)')
    parser.add_argument("--n_samples", type = int, default = 64)
    parser.add_argument("--tau", type = float, default = 0.5)
    parser.add_argument("--critic", type = str, default = 'l1')
    parser.add_argument("--patch_size", type = int, default = 128)
    parser.add_argument("--is_msk", type = int, default = 0)
    parser.add_argument("--is_bi", type = int, default = 0)
    parser.add_argument("--is_stacked_pixel_nce", type = int, default = 1)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['img_size'] = eval(opt['img_size'])

    print('n_samples: %d, patch_size: %d, is_msk: %d, crtic: %s, is_stacked_pixel_nce: %d' % (opt['n_samples'], opt['patch_size'], opt['is_msk'], opt['critic'], opt['is_stacked_pixel_nce']))

    run(opt)

'''
python train_ShgBfReg_infonceAndXFeat.py -m tiramisuAndXfeatComplex111Msk1Ps128 -bs 1 --gpu_id 0 ti_pretrained=1 enable_grad_xfeat=1 xf_pretrained=1 --is_msk 1 --patch_size 128
python train_ShgBfReg_infonceAndXFeat.py -m tiramisuAndXfeatComplex001Msk1Ps128 -bs 1 --gpu_id 0 ti_pretrained=0 enable_grad_xfeat=1 xf_pretrained=1 --is_msk 1 --patch_size 128
'''