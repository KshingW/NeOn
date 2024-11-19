import re
import os
import cv2
import glob
import scipy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from collections import deque, OrderedDict
from utils.surface_distance import compute_robust_hausdorff, compute_surface_distances
from sklearn.cluster import MiniBatchKMeans
VALUE_TYPE = torch.float32

def relabel_clusters(clusters):
    k = clusters.shape[0]
    ch = clusters.shape[1]

    def dist(a, b):
        return np.sqrt(np.sum(np.square(a-b)))

    # used will be a list of tuples of (centroid_sum, n)
    used = []
    unused = list(range(k))
    prev_cluster = np.zeros((ch,))
    while len(unused) > 0:
        best_i = 0
        best_d = dist(prev_cluster, clusters[unused[0], :])
        for i in range(1, len(unused)):
            d = dist(prev_cluster, clusters[unused[i], :])
            if d < best_d:
                best_d = d
                best_i = i
        prev_cluster = clusters[unused[best_i], :]
        used.append(unused[best_i])
        unused.pop(best_i)
    return np.array(used)

def to_tensor(A, on_gpu=True):
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(A_tensor, (1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2]))
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=VALUE_TYPE), on_gpu=on_gpu)


def image2cat_kmeans(I, k, batch_size=100, max_iter=1000, random_seed=1000):
    total_shape = I.shape
    spatial_shape = total_shape[:-1]
    channels = total_shape[-1]
    if k == 1:
        return np.zeros(spatial_shape, dtype='int')
    I_lin = I.reshape(-1, channels)
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter=max_iter, batch_size=batch_size, random_state=random_seed).fit(
        I_lin)
    centers = kmeans.cluster_centers_

    I_res = kmeans.labels_

    labs = relabel_clusters(centers)
    I_res = labs[I_res]

    return I_res.reshape(spatial_shape)

def get_downsampled_images(img, n_downs=4, mode='bilinear', n_cs=1):

    if n_cs > 0:
        blur = GaussianBlur2D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        if n_cs > 0:
            img = blur(img)
        if mode == 'nearest':
            img = F.interpolate(img, scale_factor=0.5, mode=mode)
        else:
            img = F.interpolate(img, scale_factor=0.5, mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, is_grid_out=False, mode=None, align_corners=True, padding_mode='zeros'):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mode is None:
            mode = self.mode

        out = F.grid_sample(src,new_locs,align_corners=align_corners,mode=mode,padding_mode=padding_mode)

        if is_grid_out:
            return out, new_locs
        return out

class registerSTModel(nn.Module):

    def __init__(self, img_size=(64, 256, 256), mode='bilinear', padding_mode='zeros', align_corners=True):
        super(registerSTModel, self).__init__()

        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, img, flow, is_grid_out=False, mode=None, align_corners=True, padding_mode='zeros'):

        out = self.spatial_trans(img,flow,is_grid_out,mode,align_corners,padding_mode)

        return out

class VecInt(nn.Module):

    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps

        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):

        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)

        return vec

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
    x = param_group['lr']
    return x

def dice_eval(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_cls)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_cls)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)

    dscs = []
    if output_individual:
        dscs = [torch.mean(dsc[:,x:x+1]) for x in range(1,num_cls)]

    if exclude_background:
        out = [torch.mean(torch.mean(dsc[:,1:], dim=1))] + dscs
    else:
        out = [torch.mean(torch.mean(dsc, dim=1))] + dscs

    if len(out) == 1:
        return out[0]
    return tuple(out)

def dice_eval_2D(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_cls)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 3, 1, 2).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_cls)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 3, 1, 2).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3])
    union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
    dsc = (2.*intersection) / (union + 1e-5)

    dscs = []
    if output_individual:
        dscs = [torch.mean(dsc[:,x:x+1]) for x in range(1,num_cls)]

    if exclude_background:
        out = [torch.mean(torch.mean(dsc[:,1:], dim=1))] + dscs
    else:
        out = [torch.mean(torch.mean(dsc, dim=1))] + dscs

    if len(out) == 1:
        return out[0]
    return tuple(out)


def convert_pytorch_grid2scipy(grid):

    _, H, W, D = grid.shape
    grid_x = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2

    grid = np.stack([grid_z, grid_y, grid_x])

    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid

    return grid

# def dice_missing_eval(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

#     for i in range(1, num_cls):
#         if torch.sum(y_true == i) == 0:
#             y_true = y_true + (y_pred == i).float()

def dice_binary(pred, truth, k = 1):
    truth[truth!=k]=0
    pred[pred!=k]=0
    truth=truth/k
    pred=pred/k
    intersection = np.sum(pred[truth==1.0]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(truth)+1e-7)

    return dice

def compute_tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)


class modelSaver():

    def __init__(self, save_path, save_freq, n_checkpoints = 10):

        self.save_path = save_path
        self.save_freq = save_freq
        self.best_score = -1e6
        self.best_loss = 1e6
        self.n_checkpoints = n_checkpoints
        self.epoch_fifos = deque([])
        self.score_fifos = deque([])
        self.loss_fifos = deque([])

        self.initModelFifos()

    def initModelFifos(self):

        epoch_epochs = []
        score_epochs = []
        loss_epochs  = []

        file_list = glob.glob(os.path.join(self.save_path, '*epoch*.pth'))
        if file_list:
            for file_ in file_list:
                file_name = "net_epoch_(.*)_score_.*.pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    epoch_epochs.append(int(result[0]))

                file_name = "best_score_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    score_epochs.append(int(result[0]))

                file_name = "best_loss_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    loss_epochs.append(int(result[0]))

        score_epochs.sort()
        epoch_epochs.sort()
        loss_epochs.sort()

        if file_list:
            for file_ in file_list:
                for epoch_epoch in epoch_epochs:
                    file_name = "net_epoch_" + str(epoch_epoch) + "_score_.*.pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.epoch_fifos.append(result[0])

                for score_epoch in score_epochs:
                    file_name = "best_score_.*_net_epoch_" + str(score_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.score_fifos.append(result[0])

                for loss_epoch in loss_epochs:
                    file_name = "best_loss_.*_net_epoch_" + str(loss_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.loss_fifos.append(result[0])

        print("----->>>> BEFORE: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

        self.updateFIFOs()

        print("----->>>> AFTER: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

    def saveModel(self, model, epoch, avg_score, loss=None):

        torch.save(model.state_dict(), os.path.join(self.save_path, 'net_latest.pth'))

        if epoch % self.save_freq == 0:

            file_name = ('net_epoch_%d_score_%.4f.pth' % (epoch, avg_score))
            self.epoch_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if avg_score >= self.best_score:

            self.best_score = avg_score
            file_name = ('best_score_%.4f_net_epoch_%d.pth' % (avg_score, epoch))
            self.score_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if loss is not None and loss <= self.best_loss:

            self.best_loss = loss
            file_name = ('best_loss_%.4f_net_epoch_%d.pth' % (loss, epoch))
            self.loss_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        self.updateFIFOs()

    def updateFIFOs(self):

        while(len(self.epoch_fifos) > self.n_checkpoints):
            file_name = self.epoch_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.score_fifos) > self.n_checkpoints):
            file_name = self.score_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.loss_fifos) > self.n_checkpoints):
            file_name = self.loss_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

def convert_state_dict(state_dict, is_multi = False):

    new_state_dict = OrderedDict()

    if is_multi:
        if next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is a DataParallel model_state

        for k, v in state_dict.items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
    else:

        if not next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is not a DataParallel model_state

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

    return new_state_dict

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

    return jacdet

def jacobian_determinant_2d(disp):
    # Assuming disp has shape [2, H, W], representing displacement in two dimensions
    H, W = disp.shape[1], disp.shape[2]

    # Define gradients for x and y directions
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3)

    # Compute gradients of displacement components
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :], grady, mode='constant', cval=0.0)], axis=1)

    # Stack gradients to form the Jacobian matrix for each point
    grad_disp = np.concatenate([gradx_disp, grady_disp], 0)
    # Add identity matrix since the displacement is relative to an identity grid
    jacobian = grad_disp + np.eye(2).reshape(2, 2, 1, 1)

    # Crop the edges to reduce edge effects
    # Note: Adjust this line if you need a different cropping strategy
    jacobian_cropped = jacobian[:, :, 2:-2, 2:-2]

    # Calculate determinant of the 2x2 Jacobian at each point
    jacdet = jacobian_cropped[0, 0, :, :] * jacobian_cropped[1, 1, :, :] - jacobian_cropped[0, 1, :, :] * jacobian_cropped[1, 0, :, :]

    return jacdet

def compute_HD95(moving, fixed, moving_warped,num_classes=14,spacing=np.ones(3)):

    hd95 = 0
    count = 0
    for i in range(1, num_classes):
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            continue
        if ((moving_warped==i).sum()==0):
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving==i), spacing), 95.)
        else:
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), spacing), 95.)
        count += 1
    hd95 /= count

    return hd95

def computeJacDetVal(jac_det, img_size):

    jac_det_val = np.sum(jac_det <= 0) / np.prod(img_size)

    return jac_det_val

def computeSDLogJ(jac_det, rho=3):

    log_jac_det = np.log(np.abs((jac_det+rho).clip(1e-8, 1e8)))
    std_dev_jac = np.std(log_jac_det)

    return std_dev_jac

class GaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
        super(GaussianBlur3D, self).__init__()
        self.channels = channels
        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size**2).view(kernel_size, kernel_size, kernel_size)
        y_grid = x_grid.transpose(0, 1)
        z_grid = x_grid.transpose(0, 2)
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance) ** 1.5) * \
                          torch.exp(
                              -torch.sum((xyz_grid - mean) ** 2., dim=-1) /
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv3d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred

class GaussianBlur2D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
        super(GaussianBlur2D, self).__init__()
        self.channels = channels

        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        # Create a 2D Gaussian kernel
        coord = torch.arange(kernel_size)
        grid = coord.repeat(kernel_size).view(kernel_size, kernel_size)
        xy_grid = torch.stack([grid, grid.t()], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv2d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred

class AnisotropicGaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=(1, 1, 1), kernel_size=0):
        super(AnisotropicGaussianBlur3D, self).__init__()
        self.channels = channels
        sigma_d, sigma_h, sigma_w = sigma

        if kernel_size == 0:
            kernel_size_d = int(2.0 * sigma_d * 2 + 1)
            kernel_size_h = int(2.0 * sigma_h * 2 + 1)
            kernel_size_w = int(2.0 * sigma_w * 2 + 1)
        else:
            kernel_size_d = kernel_size_h = kernel_size_w = kernel_size

        # Create a 3D Gaussian kernel for each dimension
        d_coord = torch.arange(kernel_size_d)
        h_coord = torch.arange(kernel_size_h)
        w_coord = torch.arange(kernel_size_w)

        d_grid = d_coord.repeat(kernel_size_h, kernel_size_w, 1).permute(2, 0, 1)
        h_grid = h_coord.repeat(kernel_size_d, kernel_size_w, 1).permute(1, 0, 2)
        w_grid = w_coord.repeat(kernel_size_d, kernel_size_h, 1).permute(1, 2, 0)

        mean_d = (kernel_size_d - 1) / 2.
        mean_h = (kernel_size_h - 1) / 2.
        mean_w = (kernel_size_w - 1) / 2.

        variance_d = sigma_d ** 2.
        variance_h = sigma_h ** 2.
        variance_w = sigma_w ** 2.

        # Calculate the Gaussian kernel
        gaussian_kernel = (1. / ((2. * np.pi) ** 1.5 * sigma_d * sigma_h * sigma_w)) * \
                          torch.exp(-(((d_grid - mean_d) ** 2.) / (2 * variance_d) +
                                      ((h_grid - mean_h) ** 2.) / (2 * variance_h) +
                                      ((w_grid - mean_w) ** 2.) / (2 * variance_w)))

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 3d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size_d, kernel_size_h, kernel_size_w)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding_d = kernel_size_d // 2
        self.padding_h = kernel_size_h // 2
        self.padding_w = kernel_size_w // 2

    def forward(self, x):
        x = F.pad(x, (self.padding_w, self.padding_w, self.padding_h, self.padding_h, self.padding_d, self.padding_d), mode='replicate')
        blurred = F.conv3d(x, self.gaussian_kernel, groups=self.channels)
        return blurred

def sliding_window_inference(model, input_image, roi_size=(224,224), overlap=0.75, scale_factor=14, n_cs=1280):

    b, _, h, w = input_image.shape

    h1, w1 = roi_size
    stride_h = int((1 - overlap) * h1)
    stride_w = int((1 - overlap) * w1)
    h2, w2 = h1 // scale_factor, w1 // scale_factor

    pad_h = (h1 - h % h1) % h1
    pad_w = (w1 - w % w1) % w1
    input_image = F.pad(input_image, (0, pad_w, 0, pad_h), mode='constant', value=0)

    _, _, padded_h, padded_w = input_image.shape

    output_feature_map = torch.zeros((b, n_cs, (padded_h // scale_factor), (padded_w // scale_factor)), device=input_image.device)

    for i in range(0, padded_h - h1 + 1, stride_h):
        for j in range(0, padded_w - w1 + 1, stride_w):
            roi = input_image[:, :, i:i+h1, j:j+w1]
            output = model(roi)
            output = output.view(1,h2,w2,-1).permute(0,3,1,2).contiguous()
            assert output.shape[2:] == (h2, w2), "Model output size is incorrect."

            output_h_start = i // scale_factor
            output_w_start = j // scale_factor

            output_feature_map[:, :, output_h_start:output_h_start + h2, output_w_start:output_w_start + w2] = output

    return output_feature_map


def convert_pytorch_grid2flow(grid, return_identity=False):

    grid = grid.permute(0,3,1,2)

    b,_, H, W = grid.shape
    grid_x = (grid[:,0:1, ...] + 1) * (W -1)/2
    grid_y = (grid[:,1:2, ...] + 1) * (H -1)/2

    grid = torch.cat([grid_y, grid_x],dim=1)
    identity_grid = torch.stack(torch.meshgrid([torch.arange(H), torch.arange(W)])).to(grid.device).float().unsqueeze(0)
    flow = grid - identity_grid

    if return_identity:
        return flow, identity_grid
    return flow

class MatrixKeypointAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def get_matrix(self, p1, p2, w=None):
        pass

    def forward(self, *args, **kwargs):
        return self.grid_from_points(*args, **kwargs)

    def grid_from_points(
        self,
        points_m,
        points_f,
        grid_shape,
        weights=None,
        lmbda=None,
        compute_on_subgrids=False,
    ):
        # Note we flip the order of the points here
        matrix = self.get_matrix(points_f, points_m, w=weights)
        grid = F.affine_grid(matrix, grid_shape, align_corners=False)
        return grid

    def flow_from_points(self, moving_points, fixed_points, grid_shape, weights=None, **kwargs):
        grid = self.grid_from_points(moving_points, fixed_points, grid_shape, weights, **kwargs)
        flow = convert_pytorch_grid2flow(grid)
        return flow

    def deform_points(self, points, matrix):
        square_mat = torch.zeros(len(points), self.dim + 1, self.dim + 1).to(
            points.device
        )
        square_mat[:, : self.dim, : self.dim + 1] = matrix
        square_mat[:, -1, -1] = 1
        batch_size, num_points, _ = points.shape

        points = torch.cat(
            (points, torch.ones(batch_size, num_points, 1).to(points.device)), dim=-1
        )
        warp_points = torch.bmm(square_mat[:, :3, :], points.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        return warp_points

    def points_from_points(
        self, moving_points, fixed_points, points, weights=None, **kwargs
    ):
        affine_matrix = self.get_matrix(moving_points, fixed_points, w=weights)
        square_mat = torch.zeros(len(points), self.dim + 1, self.dim + 1).to(
            moving_points.device
        )
        square_mat[:, : self.dim, : self.dim + 1] = affine_matrix
        square_mat[:, -1, -1] = 1
        batch_size, num_points, _ = points.shape

        points = torch.cat(
            (points, torch.ones(batch_size, num_points, 1).to(moving_points.device)),
            dim=-1,
        )
        warped_points = torch.bmm(
            square_mat[:, :3, :], points.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return warped_points


class AffineKeypointAligner(MatrixKeypointAligner):
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim

    def get_matrix(self, x, y, w=None):
        """
        Find A which is the solution to argmin_A \sum_i ||y_i - Ax_i||_2 = argmin_A ||Ax - y||_F
        Computes the closed-form affine equation: A = y x^T (x x^T)^(-1).

        If w provided, solves the weighted affine equation:
          A = y diag(w) x^T  (x diag(w) x^T)^(-1).
          See https://www.wikiwand.com/en/Weighted_least_squares.

        Args:
          x, y: [n_batch, n_points, dim]
          w: [n_batch, n_points]
        Returns:
          A: [n_batch, dim, dim+1]
        """
        # Take transpose as columns should be the points
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        # Convert y to homogenous coordinates
        one = torch.ones(x.shape[0], 1, x.shape[2]).float().to(x.device)
        x = torch.cat([x, one], 1)

        out = torch.bmm(x, torch.transpose(x, -2, -1))
        inv = torch.inverse(out)
        out = torch.bmm(torch.transpose(x, -2, -1), inv)
        out = torch.bmm(y, out)

        return out



def extract_pixel_features(x, ks=2):

    b,_,h,w = x.shape

    x = F.pad(x, (ks,)*2*2)
    x = x.unfold(2, ks*2+1, 1).unfold(3, ks*2+1, 1)
    x = x.contiguous().view(b,h,w,-1).permute(0,3,1,2)

    return x

def get_matrix(x, y):
    """
    Find A which is the solution to argmin_A \sum_i ||y_i - Ax_i||_2 = argmin_A ||Ax - y||_F
    Computes the closed-form affine equation: A = y x^T (x x^T)^(-1).

    If w provided, solves the weighted affine equation:
        A = y diag(w) x^T  (x diag(w) x^T)^(-1).
        See https://www.wikiwand.com/en/Weighted_least_squares.

    Args:
        x, y: [n_batch, n_points, dim]
        w: [n_batch, n_points]
    Returns:
        A: [n_batch, dim, dim+1]
    """
    # Take transpose as columns should be the points
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)

    # Convert y to homogenous coordinates
    one = torch.ones(x.shape[0], 1, x.shape[2]).float().to(x.device)
    x = torch.cat([x, one], 1)

    out = torch.bmm(x, torch.transpose(x, -2, -1))
    inv = torch.inverse(out)
    out = torch.bmm(torch.transpose(x, -2, -1), inv)
    out = torch.bmm(y, out)

    return out

def plot_images_with_keypoints(img_x, img_y, mkpts_0, mkpts_1):
    # Convert images to numpy for plotting.
    img_x_np = img_x.squeeze().numpy()  # Removing unnecessary dimensions
    img_y_np = img_y.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Setup a figure with two subplots

    # Plotting the first image and its keypoints
    axes[0].imshow(img_x_np, cmap='gray')
    axes[0].scatter(mkpts_0[:, 0], mkpts_0[:, 1], color='red', s=10)  # Plot keypoints as red dots
    axes[0].set_title('Image X with keypoints')
    axes[0].axis('off')  # Hide axes

    # Plotting the second image and its keypoints
    axes[1].imshow(img_y_np, cmap='gray')
    axes[1].scatter(mkpts_1[:, 0], mkpts_1[:, 1], color='blue', s=10)  # Plot keypoints as blue dots
    axes[1].set_title('Image Y with keypoints')
    axes[1].axis('off')  # Hide axes

    plt.show()

def plot_images_with_keypoints_and_lines(img_x, img_y, mkpts_0, mkpts_1):
    img_x_np = img_x.squeeze().numpy()
    img_y_np = img_y.squeeze().numpy()

    # Setup figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Display both images side by side
    ax.imshow(np.concatenate((img_x_np, img_y_np), axis=1), cmap='gray')
    ax.axis('off')

    # Adjust mkpts_1 x-coordinates for the offset due to image concatenation
    mkpts_1_adjusted = mkpts_1.copy()
    mkpts_1_adjusted[:, 0] += img_x_np.shape[1]  # Offset by the width of the first image

    # Plot keypoints
    ax.scatter(mkpts_0[:, 0], mkpts_0[:, 1], color='red', s=10, label='Keypoints in Image X')
    ax.scatter(mkpts_1_adjusted[:, 0], mkpts_1_adjusted[:, 1], color='blue', s=10, label='Keypoints in Image Y')

    # Draw lines between matching keypoints
    for (x0, y0), (x1, y1) in zip(mkpts_0, mkpts_1_adjusted):
        ax.plot([x0, x1], [y0, y1], 'y-', linewidth=0.5)  # Yellow lines

    plt.legend(loc='upper right')
    plt.show()

def find_matches(ref_points, dst_points):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Filter points based on the homography mask
    filtered_ref_points = ref_points[mask == 1]
    filtered_dst_points = dst_points[mask == 1]

    return filtered_ref_points, filtered_dst_points

def normalize_common(a, b):
    if isinstance(a, torch.Tensor):
        a = tensor2np(a)
    if isinstance(b, torch.Tensor):
        b = tensor2np(b)
    a_min, a_max = np.percentile(a, [1, 99])
    b_min, b_max = np.percentile(b, [1, 99])
    minimum = max(a_min, b_min)
    maximum = min(a_max, b_max)
    new_a = np.clip((a - minimum) / (maximum - minimum), 0, 1)
    new_b = np.clip((b - minimum) / (maximum - minimum), 0, 1)
    return new_a, new_b

def normalize_common_torch(a, b):
    a_min, a_max = torch.kthvalue(a.view(-1), int(0.01 * a.numel()))[0], torch.kthvalue(a.view(-1), int(0.99 * a.numel()))[0]
    b_min, b_max = torch.kthvalue(b.view(-1), int(0.01 * b.numel()))[0], torch.kthvalue(b.view(-1), int(0.99 * b.numel()))[0]
    minimum = max(a_min, b_min)
    maximum = min(a_max, b_max)
    new_a = torch.clamp((a - minimum) / (maximum - minimum), 0, 1)
    new_b = torch.clamp((b - minimum) / (maximum - minimum), 0, 1)

    return new_a, new_b