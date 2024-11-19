import math
import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F



class DiceLoss(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=14, is_square=False):
        super().__init__()
        self.num_class = num_class
        self.is_square = is_square

    def forward(self, y_pred, y_true):
        '''
        Assuming y_pred has been one-hot encoded: [bs, num_class, h, w, d]
        '''
        y_true = nn.functional.one_hot(y_true.long(), num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()

        if y_pred.shape[2] != y_true.shape[2] or y_pred.shape[3] != y_true.shape[3] or y_pred.shape[4] != y_true.shape[4]:
            y_pred = nn.functional.interpolate(y_pred, size=(y_true.shape[2], y_true.shape[3], y_true.shape[4]), mode='trilinear', align_corners=True)

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        if self.is_square:
            union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        else:
            union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class DiceLoss2D(nn.Module):
    """Dice loss"""

    def __init__(self, num_class=14, is_square=False):
        super().__init__()
        self.num_class = num_class
        self.is_square = is_square

    def forward(self, y_pred, y_true):
        '''
        Assuming y_pred has been one-hot encoded: [bs, num_class, h, w, d]
        '''
        y_true = nn.functional.one_hot(y_true.long(), num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 3, 1, 2).contiguous()

        if y_pred.shape[2] != y_true.shape[2] or y_pred.shape[3] != y_true.shape[3]:
            y_pred = nn.functional.interpolate(y_pred, size=(y_true.shape[2], y_true.shape[3]), mode='bilinear', align_corners=True)

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3])

        if self.is_square:
            union = torch.pow(y_pred, 2).sum(dim=[2, 3]) + torch.pow(y_true, 2).sum(dim=[2, 3])
        else:
            union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class BinaryDiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):

        intersection = y_pred * y_true
        intersection = intersection.sum(dim=(2,3,4))
        union = y_pred.sum(dim=(2,3,4)) + y_true.sum(dim=(2,3,4))
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))

        return dsc

class Grad3d(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1'):
        super(Grad3d, self).__init__()

        self.penalty = penalty

    def forward(self, y_pred, y_true=None):

        dy = ((y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])**2).mean()
        dx = ((y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])**2).mean()
        dz = ((y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])**2).mean()
        grad = (dy + dx + dz) / 3.0

        return grad

class Grad2d(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1'):
        super(Grad2d, self).__init__()

        self.penalty = penalty

    def forward(self, y_pred, y_true=None):

        dy = ((y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])**2).mean()
        dx = ((y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])**2).mean()
        grad = (dy + dx) / 2.0

        return grad

class NccLossBak(nn.Module):

    def __init__(self, win=None):
        super(NccLossBak, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        c = y_true.shape[1]
        # compute filters
        sum_filt = torch.ones([c, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding, groups=c)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding, groups=c)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding, groups=c)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding, groups=c)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding, groups=c)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return (1.-torch.mean(cc, dim=[2,3,4])).mean() # this seems not correct

class NccLoss(nn.Module):

    def __init__(self, win=None):
        super(NccLoss, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1.-torch.mean(cc)

def random_pixels(x, y, n, ref=None):
    b, c, h, w = x.shape
    device = x.device

    x_flat = x.view(b, c, h * w)
    y_flat = y.view(b, c, h * w)

    if ref is None:
        msk = torch.ones(b, h * w, device=device).bool()
    else:
        ref_flat = ref.view(b, h * w)
        msk = ref_flat > 0.001

    # Initialize containers for the randomly chosen pixels
    x_rand = torch.empty(b, c, n, device=device)
    y_rand = torch.empty(b, c, n, device=device)

    # For each example in the batch, select n indices where the condition is met
    for i in range(b):
        valid_indices = msk[i].nonzero(as_tuple=True)[0]
        num_valid = len(valid_indices)

        if num_valid == 0:
            raise ValueError("No pixels above threshold to sample from in batch item {}".format(i))

        # If fewer valid indices than n, use all valid indices; otherwise, randomly sample
        if num_valid < n:
            chosen_indices = valid_indices
        else:
            chosen_indices = valid_indices[torch.randint(0, num_valid, (n,))]

        # Ensure that the number of chosen pixels matches n, repeat indices if necessary
        if len(chosen_indices) < n:
            repeat_times = (n + num_valid - 1) // num_valid  # Calculate how many times to repeat array
            chosen_indices = chosen_indices.repeat(repeat_times)[:n]

        # Gather the pixels based on the chosen indices
        chosen_indices = chosen_indices.expand(c, -1)  # Expand indices to cover all channels
        x_rand[i] = torch.gather(x_flat[i], 1, chosen_indices)
        y_rand[i] = torch.gather(y_flat[i], 1, chosen_indices)

    return x_rand, y_rand  # [b, c, n]

def random_patches(x, y, n, patch_size):
    b, c, h, w = x.shape
    device = x.device

    # Adjust the range of the random indices to ensure patches don't cross the image borders
    h_range = h - patch_size + 1
    w_range = w - patch_size + 1

    # Generate random starting points for the patches
    start_rows = torch.randint(0, h_range, (b, n), device=device)
    start_cols = torch.randint(0, w_range, (b, n), device=device)

    # Initialize the output tensors for x and y
    x_patches = torch.zeros(b, n, c, patch_size, patch_size, device=device)
    y_patches = torch.zeros(b, n, c, patch_size, patch_size, device=device)

    # Extract patches
    for i in range(n):
        for j in range(b):
            x_patches[j, i] = x[j, :, start_rows[j, i]:start_rows[j, i]+patch_size, start_cols[j, i]:start_cols[j, i]+patch_size]
            y_patches[j, i] = y[j, :, start_rows[j, i]:start_rows[j, i]+patch_size, start_cols[j, i]:start_cols[j, i]+patch_size]

    x_patches = x_patches.view(b, n, -1)
    y_patches = y_patches.view(b, n, -1)

    return x_patches, y_patches  # [b, n, c, patch_size, patch_size]

def random_patches_msk(x, y, n, patch_size, ref=None):
    b, c1, h, w = x.shape
    b, c2, h, w = y.shape
    device = x.device

    # Adjust the range of the random indices to ensure patches don't cross the image borders
    h_range = h - patch_size + 1
    w_range = w - patch_size + 1

    # Prepare mask based on ref if provided
    if ref is not None:
        valid_mask = F.avg_pool2d(ref.float(), patch_size, stride=1, padding=0) > 0
    else:
        valid_mask = torch.ones(b, h_range, w_range, dtype=torch.bool, device=device)

    # Generate random starting points for the patches from valid positions
    x_patches = torch.zeros(b, n, c1, patch_size, patch_size, device=device)
    y_patches = torch.zeros(b, n, c2, patch_size, patch_size, device=device)

    for i in range(b):
        valid_indices = valid_mask[i].nonzero(as_tuple=True)
        num_valid = len(valid_indices[0])

        if num_valid == 0:
            raise ValueError("No valid patches above threshold to sample from in batch item {}".format(i))

        # Random sampling of indices
        # chosen_indices = torch.randint(0, num_valid, (n,), device=device)
        chosen_indices = torch.randperm(num_valid,device=device)[:n]

        for j in range(n):
            row_idx = valid_indices[0][chosen_indices[j]]
            col_idx = valid_indices[1][chosen_indices[j]]
            x_patches[i, j] = x[i, :, row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]
            y_patches[i, j] = y[i, :, row_idx:row_idx+patch_size, col_idx:col_idx+patch_size]

    return x_patches, y_patches  # [b, n, c, patch_size, patch_size]

def batched_NCE_loss(x, y, tau=0.5, crtic='l2'):
    '''
    x, y : (b,c,n)
    x and y <-> x and y
    '''
    b,c,N = x.shape
    device = x.device

    pts = torch.cat([x, y], dim=2)

    if crtic == 'l2':
        similarities = -(pts.unsqueeze(3) - pts.unsqueeze(2)).pow(2).mean(dim=1) / tau
    elif crtic == 'l1':
        similarities = -(pts.unsqueeze(3) - pts.unsqueeze(2)).abs().mean(dim=1) / tau
    else:
        raise ValueError("Invalid critic type")
    similarities = (similarities + similarities.transpose(1, 2)) / 2

    # similarities = torch.zeros(b, 2*N, 2*N, device=device)
    # for i in range(2*N):
    #     for j in range(i+1):
    #         s = -(pts[:, :, i] - pts[:, :, j]).abs().mean(dim=1) / tau
    #         similarities[:, i, j] = s
    #         similarities[:, j, i] = s

    # Compute indices for positive samples
    irange = torch.arange(2*N, device=device)
    j_indices = (irange + N) % (2 * N)
    j_indices = j_indices.unsqueeze(0).expand(b, -1).unsqueeze(-1)
    pos = torch.gather(similarities, 2, j_indices).squeeze(-1) # (b, 2*N)

    irange_expand = irange.unsqueeze(0).expand(2*N, 2*N) # ==irange.expand(2*N, 2*N)
    msk = irange_expand != irange.view(-1, 1) # (2*N, 2*N)

    neg_entries = similarities[:,msk].view(b, 2*N, 2*N-1)
    neg = torch.logsumexp(neg_entries, dim=-1) # (b, 2*N)

    softmaxes = -pos + neg

    return softmaxes

def batched_NCE_half_loss(x, y, tau=0.5, crtic='l2', is_loop=False):
    '''
    x, y : (b,c,n)
    x <-> x and y
    '''
    b,c,N = x.shape
    device = x.device

    pts = torch.cat([x, y], dim=2)

    if is_loop:
        similarities = torch.zeros(b, 2*N, 2*N, device=device)
        for i in range(2*N):
            for j in range(i+1):
                if crtic == 'l2':
                    s = -(pts[:, :, i] - pts[:, :, j]).pow(2).mean(dim=1) / tau
                elif crtic == 'l1':
                    s = -(pts[:, :, i] - pts[:, :, j]).abs().mean(dim=1) / tau
                similarities[:, i, j] = s
                similarities[:, j, i] = s
    else:
        if crtic == 'l2':
            similarities = -(x.unsqueeze(3) - pts.unsqueeze(2)).pow(2).mean(dim=1) / tau
        elif crtic == 'l1':
            similarities = -(x.unsqueeze(3) - pts.unsqueeze(2)).abs().mean(dim=1) / tau
        else:
            raise ValueError("Invalid critic type")
    print('similarities', similarities.shape)

    irange = torch.arange(N, device=device)
    j_indices = (irange + N)
    j_indices = j_indices.unsqueeze(0).expand(b, -1).unsqueeze(-1) # (b, N, 1)
    pos = torch.gather(similarities, 2, j_indices).squeeze(-1) # (b, N)

    self_j_indices = irange.unsqueeze(0).expand(b, -1).unsqueeze(-1) # (b, N, 1)
    pos_self = torch.gather(similarities, 2, self_j_indices).squeeze(-1) # (b, N)

    neg_entries = torch.exp(similarities).sum(dim=2) - torch.exp(pos_self) # (b, N)
    neg_entries = torch.clamp(neg_entries, min=1e-3)
    neg = torch.log(neg_entries) # (b, N)

    softmaxes = -pos + neg

    return softmaxes

def activation_decay(tensors, p=2., device=None):
    """Computes the L_p^p norm over an activation map.
    """
    if not isinstance(tensors, list):
        tensors = [tensors]
    loss = torch.tensor(1.0, device=device)
    Z = 0
    for tensor in tensors:
        Z += tensor.numel()
        loss += torch.sum(tensor.pow(p).abs()).to(device)
    return loss / Z