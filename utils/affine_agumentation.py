import numpy as np
import torch
import torch.nn.functional as F

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

class AffineDeformation2d:
    def __init__(self, device="cuda:0"):
        self.device = device

    def build_affine_matrix_2d(self, batch_size, params):
        """
        Return a affine transformation matrix
        size: size of input .size() method
        params: tuple of (s, o, a, z), where:
          s: sample scales  (bs, 2)
          o: sample offsets (bs, 2)
          a: sample angles  (bs, 1)
          z: sample shear   (bs, 2)
        """
        scale, offset, theta, shear = params
        ones = torch.ones(batch_size).float().to(self.device)

        # Scale
        Ms = torch.zeros([batch_size, 3, 3], device=self.device)
        Ms[:, 0, 0] = scale[:, 0]
        Ms[:, 1, 1] = scale[:, 1]
        Ms[:, 2, 2] = ones

        # Translation
        Mt = torch.zeros([batch_size, 3, 3], device=self.device)
        Mt[:, 0, 2] = offset[:, 0]
        Mt[:, 1, 2] = offset[:, 1]
        Mt[:, 0, 0] = ones
        Mt[:, 1, 1] = ones
        Mt[:, 2, 2] = ones

        # Rotation
        Mr = torch.zeros([batch_size, 3, 3], device=self.device)

        Mr[:, 0, 0] = torch.cos(theta[:, 0])
        Mr[:, 0, 1] = -torch.sin(theta[:, 0])
        Mr[:, 1, 0] = torch.sin(theta[:, 0])
        Mr[:, 1, 1] = torch.cos(theta[:, 0])
        Mr[:, 2, 2] = ones

        # Shear
        Mz = torch.zeros([batch_size, 3, 3], device=self.device)

        Mz[:, 0, 1] = shear[:, 0]
        Mz[:, 1, 0] = shear[:, 1]
        Mz[:, 0, 0] = ones
        Mz[:, 1, 1] = ones
        Mz[:, 2, 2] = ones

        M = torch.bmm(Mz, torch.bmm(Ms, torch.bmm(Mt, Mr)))
        return M

    def deform_img(self, img, params, interp_mode="bilinear", is_flow_output=False):
        Ma = self.build_affine_matrix_2d(len(img), params)
        phi_inv = F.affine_grid(
            torch.inverse(Ma)[:, :2, :], img.size(), align_corners=False
        ).to(self.device)
        neg_flow = convert_pytorch_grid2flow(phi_inv)
        img_moved = F.grid_sample(
            img,
            grid=phi_inv, 
            mode=interp_mode,
            padding_mode="border", 
            align_corners=False
        )
        phi = F.affine_grid(
            Ma[:, :2, :], img.size(), align_corners=False
        ).to(self.device)
        pos_flow = convert_pytorch_grid2flow(phi)

        if is_flow_output:
            return img_moved, neg_flow, pos_flow
        return img_moved

    def deform_points(self, points, params):
        batch_size, num_points, dim = points.shape
        Ma = self.build_affine_matrix_2d(batch_size, params)
        points = torch.cat(
            (points, torch.ones(batch_size, num_points, 1).to(self.device)), dim=-1
        )
        warp_points = torch.bmm(Ma[:, :2, :], points.permute(0, 2, 1)).permute(0, 2, 1)
        return warp_points

    def __call__(self, img, **kwargs):
        params = kwargs["params"]
        interp_mode = kwargs["interp_mode"]
        is_flow_output = kwargs["is_flow_output"]
        return self.deform_img(img, params, interp_mode, is_flow_output)

class AffineDeformation3d:
    def __init__(self, device="cuda:0"):
        self.device = device

    def build_affine_matrix_3d(self, batch_size, params):
        """
        Return a affine transformation matrix
        batch_size: size of batch
        params: tuple of torch.FloatTensor
          scales  (batch_size, 3)
          offsets (batch_size, 3)
          angles  (batch_size, 3)
          shear   (batch_size, 6)
        """
        scale, offset, theta, shear = params

        ones = torch.ones(batch_size).float().to(self.device)

        # Scaling
        Ms = torch.zeros([batch_size, 4, 4], device=self.device)
        Ms[:, 0, 0] = scale[:, 0]
        Ms[:, 1, 1] = scale[:, 1]
        Ms[:, 2, 2] = scale[:, 2]
        Ms[:, 3, 3] = ones

        # Translation
        Mt = torch.zeros([batch_size, 4, 4], device=self.device)
        Mt[:, 0, 3] = offset[:, 0]
        Mt[:, 1, 3] = offset[:, 1]
        Mt[:, 2, 3] = offset[:, 2]
        Mt[:, 0, 0] = ones
        Mt[:, 1, 1] = ones
        Mt[:, 2, 2] = ones
        Mt[:, 3, 3] = ones

        # Rotation
        dim1_matrix = torch.zeros([batch_size, 4, 4], device=self.device)
        dim2_matrix = torch.zeros([batch_size, 4, 4], device=self.device)
        dim3_matrix = torch.zeros([batch_size, 4, 4], device=self.device)

        dim1_matrix[:, 0, 0] = ones
        dim1_matrix[:, 1, 1] = torch.cos(theta[:, 0])
        dim1_matrix[:, 1, 2] = -torch.sin(theta[:, 0])
        dim1_matrix[:, 2, 1] = torch.sin(theta[:, 0])
        dim1_matrix[:, 2, 2] = torch.cos(theta[:, 0])
        dim1_matrix[:, 3, 3] = ones

        dim2_matrix[:, 0, 0] = torch.cos(theta[:, 1])
        dim2_matrix[:, 0, 2] = torch.sin(theta[:, 1])
        dim2_matrix[:, 1, 1] = ones
        dim2_matrix[:, 2, 0] = -torch.sin(theta[:, 1])
        dim2_matrix[:, 2, 2] = torch.cos(theta[:, 1])
        dim2_matrix[:, 3, 3] = ones

        dim3_matrix[:, 0, 0] = torch.cos(theta[:, 2])
        dim3_matrix[:, 0, 1] = -torch.sin(theta[:, 2])
        dim3_matrix[:, 1, 0] = torch.sin(theta[:, 2])
        dim3_matrix[:, 1, 1] = torch.cos(theta[:, 2])
        dim3_matrix[:, 2, 2] = ones
        dim3_matrix[:, 3, 3] = ones

        """Shear"""
        Mz = torch.zeros([batch_size, 4, 4], device=self.device)

        Mz[:, 0, 1] = shear[:, 0]
        Mz[:, 0, 2] = shear[:, 1]
        Mz[:, 1, 0] = shear[:, 2]
        Mz[:, 1, 2] = shear[:, 3]
        Mz[:, 2, 0] = shear[:, 4]
        Mz[:, 2, 1] = shear[:, 5]
        Mz[:, 0, 0] = ones
        Mz[:, 1, 1] = ones
        Mz[:, 2, 2] = ones
        Mz[:, 3, 3] = ones

        Mr = torch.bmm(dim3_matrix, torch.bmm(dim2_matrix, dim1_matrix))
        M = torch.bmm(Mz, torch.bmm(Ms, torch.bmm(Mt, Mr)))
        return M

    def deform_img(self, img, params, interp_mode="bilinear"):
        Ma = self.build_affine_matrix_3d(len(img), params)
        phi_inv = F.affine_grid(
            torch.inverse(Ma)[:, :3, :], img.size(), align_corners=False
        ).to(self.device)
        neg_flow = convert_pytorch_grid2flow(phi_inv)
        img_moved = F.grid_sample(
            img.to(self.device),
            grid=phi_inv,
            mode=interp_mode,
            padding_mode="border",
            align_corners=False,
        )
        phi = F.affine_grid(
            Ma[:, :3, :], img.size(), align_corners=False
        ).to(self.device)
        pos_flow = convert_pytorch_grid2flow(phi)

        return img_moved

    def deform_points(self, points, params):
        batch_size, num_points, dim = points.shape
        Ma = self.build_affine_matrix_3d(batch_size, params)
        points = torch.cat(
            (points, torch.ones(batch_size, num_points, 1).to(self.device)), dim=-1
        )
        warp_points = torch.bmm(Ma[:, :3, :], points.permute(0, 2, 1)).permute(0, 2, 1)
        return warp_points

    def __call__(self, img, **kwargs):
        params = kwargs["params"]
        interp_mode = kwargs["interp_mode"]
        return self.deform_img(img, params, interp_mode)

# Convenience functions
def random_affine_augment(
    img,
    img2=None,
    seg=None,
    seg2=None,
    points=None,
    max_random_params=(0.3, 0.3, 3.1416, 0.1),
    scale_params=None,
    is_flow_output=False,
):
    """Randomly augment moving image. Optionally augments corresponding segmentation and keypoints.

    :param img: Moving image to augment (bs, nch, l, w) or (bs, nch, l, w, h)
    :param max_random_params: 4-tuple of floats, max value of each transformation for random augmentation.
    :param scale_params: If set, scales parameters by this value. Use for ramping up degree of augmnetation.
    """
    s, o, a, z = max_random_params
    if scale_params:
        assert scale_params >= 0 and scale_params <= 1
        s *= scale_params
        o *= scale_params
        a *= scale_params
        z *= scale_params
    if len(img.shape) == 4:
        scale = torch.FloatTensor(1, 2).uniform_(1 - s, 1 + s)
        offset = torch.FloatTensor(1, 2).uniform_(-o, o)
        theta = torch.FloatTensor(1, 1).uniform_(-a, a)
        shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        augmenter = AffineDeformation2d(device=img.device)
    else:
        scale1 = torch.FloatTensor(1, 2).uniform_(1 - s, 1 + s)
        scale2 = torch.FloatTensor(1, 1).uniform_(1 - s, 1 + s)
        scale = torch.cat((scale1, scale2), dim=1)
        offset1 = torch.FloatTensor(1, 2).uniform_(-o, o)
        offset2 = torch.FloatTensor(1, 1).uniform_(-o, o)
        offset = torch.cat((offset1, offset2), dim=1)
        theta1 = torch.FloatTensor(1, 2).uniform_(-a, a)
        theta2 = torch.FloatTensor(1, 1).uniform_(-a, a)
        theta = torch.cat((theta1, theta2), dim=1)
        shear = torch.FloatTensor(1, 6).uniform_(-z, z)
        augmenter = AffineDeformation3d(device=img.device)

    params = (scale, offset, theta, shear)

    img = augmenter(img, params=params, interp_mode="bilinear", is_flow_output=is_flow_output)
    if is_flow_output:
        res = (img[0], img[1], img[2])
    else:
        res = (img, )
    if img2 is not None:
        img2 = augmenter(img2, params=params, interp_mode="bilinear", is_flow_output=False)
        res += (img2,)
    if seg is not None:
        seg = augmenter(seg, params=params, interp_mode="nearest", is_flow_output=False)
        res += (seg,)
    if seg2 is not None:
        seg2 = augmenter(seg2, params=params, interp_mode="nearest", is_flow_output=False)
        res += (seg2,)
    if points is not None:
        points = augmenter.deform_points(points, params)
        res += (points,)
    return res


def affine_augment(img, fixed_params, seg=None, points=None):
    """Augment moving image. Optionally augments corresponding segmentation and keypoints.

    :param img: Moving image to augment (bs, nch, l, w) or (bs, nch, l, w, h)
    :param fixed_params: Fixed parameters for transformation.
    """
    s, o, a, z = fixed_params
    if len(img.shape) == 4:
        scale = torch.tensor([e + 1 for e in s]).unsqueeze(0).float()
        offset = torch.tensor(o).unsqueeze(0).float()
        theta = torch.tensor(a).unsqueeze(0).float()
        shear = torch.tensor(z).unsqueeze(0).float()
        augmenter = AffineDeformation2d(device=img.device)
    else:
        scale = torch.tensor([e + 1 for e in s]).unsqueeze(0).float()
        offset = torch.tensor(o).unsqueeze(0).float()
        theta = torch.tensor(a).unsqueeze(0).float()
        shear = torch.tensor(z).unsqueeze(0).float()
        augmenter = AffineDeformation3d(device=img.device)

    params = (scale, offset, theta, shear)

    img = augmenter(img, params=params, interp_mode="bilinear")
    res = (img, )
    if seg is not None:
        seg = augmenter(seg, params=params, interp_mode="nearest")
        res += (seg,)
    if points is not None:
        points = augmenter.deform_points(points, params)
        res += (points,)

    return res

def random_affine_augment_pair(
    img1, img2, max_random_params=(0.2, 0.2, 3.1416, 0.1), scale_params=None
):
    s, o, a, z = max_random_params
    if scale_params:
        assert scale_params >= 0 and scale_params <= 1
        s *= scale_params
        o *= scale_params
        a *= scale_params
        z *= scale_params
    if len(img1.shape) == 4:
        scale = torch.FloatTensor(1, 2).uniform_(1 - s, 1 + s)
        offset = torch.FloatTensor(1, 2).uniform_(-o, o)
        theta = torch.FloatTensor(1, 1).uniform_(-a, a)
        shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        augmenter = AffineDeformation2d(device=img1.device)
    else:
        scale = torch.FloatTensor(1, 3).uniform_(1 - s, 1 + s)
        offset = torch.FloatTensor(1, 3).uniform_(-o, o)
        theta = torch.FloatTensor(1, 3).uniform_(-a, a)
        shear = torch.FloatTensor(1, 6).uniform_(-z, z)
        augmenter = AffineDeformation3d(device=img2.device)

    params = (scale, offset, theta, shear)

    img1 = augmenter(img1, params=params, interp_mode="bilinear")
    img2 = augmenter(img2, params=params, interp_mode="bilinear")
    return img1, img2

