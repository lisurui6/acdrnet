import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from backbones.unet_3d import Encoder, Decoder
# from utils.topology import get_circles, get_circles_2, get_circles_3
from utils.topology_3d_smooth_lv import sample_3d_points
import neural_renderer as nr
from torch.distributions.normal import Normal
import math
from models.layer import SpatialTransformer, VecInt, AffineSpatialTransformer
from rasterizor.voxelize import Voxelize

__all__ = ['ShapeDeformCardiac3DCircleNet']

kernel_size = 5


class ShapeDeformCardiac3DCircleNet(nn.Module):
    """No points, just mask after init. then affine and deform"""
    def __init__(self, num_nodes, enc_dim, voxel_width, voxel_height, voxel_depth, drop, num_lv_slices, num_extra_lv_myo_slices):
        super().__init__()
        self.num_nodes = num_nodes
        self.voxel_width = voxel_width
        self.voxel_height = voxel_height
        self.voxel_depth = voxel_depth
        self.num_lv_slices = num_lv_slices
        self.num_extra_lv_myo_slices = num_extra_lv_myo_slices

        padding = int((kernel_size - 1) / 2)

        self.shape_encoder = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=1)
        self.shape_regressor = nn.Sequential(
            nn.Conv3d(
                self.shape_encoder.dims[-2], self.shape_encoder.dims[-2] // 2,
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm3d(self.shape_encoder.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(self.shape_encoder.dims[-2] // 2, self.shape_encoder.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm3d(self.shape_encoder.dims[-2] // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        num_sig_dz = num_lv_slices - 1  # myo + lv

        # dz = (0, 1) * (1 - c0_z) * depth / (num_extra_lv_myo_slices + num_lv_slices)
        # dxy/dz * (n_lv + n_myo - 1), dr1/dz * (n_lv + n_myo - 1), dr0/dz * (n_lv + n_myo - 1),
        # dtheta2/dz * (n_lv + n_myo - 1), dtheta_c2/dz * (n_lv + n_myo - 1), dd_c2_c0/dz * (n_lv + n_myo - 1),
        num_tanh_delta = (num_lv_slices - 1) * 4
        self.shape_end1 = nn.Sequential(nn.Linear(200, 3), nn.Tanh())  # c0_x, c0_y and c0_z
        # bias = torch.from_numpy(np.array([0, 0, 0, 33/128])).float()
        # bias = torch.from_numpy(np.array([0, 0, -1] + [0] * num_tanh_delta)).float()
        bias = torch.from_numpy(np.array([0, 0, 0])).float()
        self.shape_end1[0].weight.data.zero_()
        self.shape_end1[0].bias.data.copy_(bias)

        self.shape_end2 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())  # r1,
        # bias = torch.from_numpy(np.array([-1.1838, 0.690, -1.6094, 0, -1.099])).float()
        bias = torch.from_numpy(np.array([-1.1838])).float()

        # self.shape_end2 = nn.Sequential(nn.Linear(200, 3), nn.Sigmoid())  # factor 0 (r0/r1), theta2/pi, d_c2_c0
        # bias = torch.from_numpy(np.array([0.690, -1.6094, 0])).float()
        self.shape_end2[0].weight.data.zero_()
        self.shape_end2[0].bias.data.copy_(bias)

        self.affine_encoder = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=4)
        self.affine_regressor = nn.Sequential(
            nn.Conv3d(
                self.affine_encoder.dims[-2], self.affine_encoder.dims[-2] // 2,
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm3d(self.affine_encoder.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(self.affine_encoder.dims[-2] // 2, self.affine_encoder.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm3d(self.affine_encoder.dims[-2] // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        self.affine_end1 = nn.Sequential(nn.Linear(200, 6), nn.Tanh())  # rotation rx, ry, rz, translation x, y, z
        bias = torch.from_numpy(np.array([0, 0, 0, 0, 0, 0])).float()
        self.affine_end1[0].weight.data.zero_()
        self.affine_end1[0].bias.data.copy_(bias)

        self.affine_end2 = nn.Sequential(nn.Linear(200, 3))  # scale x, y, z
        bias = torch.from_numpy(np.array([1, 1, 1])).float()
        self.affine_end2[0].weight.data.zero_()
        self.affine_end2[0].bias.data.copy_(bias)

        self.deform_backbone = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=4)
        self.decoder = Decoder(
            self.deform_backbone.dims, drop=drop, kernel_size=kernel_size, output_act=None,
            output_dim=self.deform_backbone.dims[-4]
        )

        self.flow = nn.Conv3d(self.deform_backbone.dims[-4], 3, kernel_size=3, padding=1)
        # # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.texture_size = 2
        self.camera_distance = 1
        self.elevation = 0
        self.azimuth = 0
        self.integrate = VecInt(
            inshape=(self.voxel_width, self.voxel_height, self.voxel_depth),
            nsteps=7,
        )
        self.tri0 = None
        self.tri1 = None
        self.tri2 = None
        self.affine_transformer = AffineSpatialTransformer(mode="bilinear")
        self.deform_transformer = SpatialTransformer(size=(self.voxel_width, self.voxel_height, self.voxel_depth), mode="bilinear")
        self.voxeliser = Voxelize(voxel_width=self.voxel_width, voxel_height=self.voxel_height, voxel_depth=self.voxel_depth)

    def forward(self, img, epoch=0, vis=False):
        # x = (B, 1, H, W)
        x = norm_tensor(img)
        out = self.shape_encoder(x)
        print(out[-1].shape)
        out = self.shape_regressor(out[-1])
        out_par1 = self.shape_end1(out)  # (B, 3), tanh
        out_par2 = self.shape_end2(out)  # (B, 5), sig
        print(out_par2[:, 0], "r1")

        nodes0, tetras0, self.tri0 = sample_3d_points(
            par1=out_par1, par2=out_par2, num_lv_slices=self.num_lv_slices,
            voxel_depth=self.voxel_depth, voxel_height=self.voxel_height, voxel_width=self.voxel_width,
            num_points=32, batch_size=img.shape[0], lv_tetras=self.tri0, lv_myo_tetras=self.tri1, rv_tetras=self.tri2,
            epoch=epoch, vis=vis,
        )
        nodes0 = torch.autograd.Variable(nodes0, requires_grad=True)
        init_mask0 = self.voxelize_mask(nodes0, tetras0)

        # plot(init_mask0)
        # plot(init_mask1)
        # plot(init_mask2)

        if self.training:
            return init_mask0, nodes0, out_par1
        else:
            return init_mask0

    def voxelize_mask(self, nodes, faces):
        P3d = torch.squeeze(nodes, dim=1)
        faces = torch.squeeze(faces, dim=1).to(nodes.device)
        mask = self.voxeliser(P3d, faces, use_cuda=True).unsqueeze(1)
        return mask


def norm_tensor(_x):
    bs = _x.shape[0]
    min_x = _x.view(bs, 1, -1).min(dim=2)[0].view(bs, 1, 1, 1, 1)
    max_x = _x.view(bs, 1, -1).max(dim=2)[0].view(bs, 1, 1, 1, 1)
    return (_x - min_x) / (max_x - min_x + 1e-2)


def affine_transform_points(points, affine_pars):
    # affine_pars = (B, 2, 3), points = (B, N, 2)
    z = torch.zeros((affine_pars.shape[0], 1, 3)).to(affine_pars.device)
    z[:, :, 2] = 1
    affine_pars = torch.cat([affine_pars, z], dim=1)
    z = torch.ones((points.shape[0], 1, points.shape[2], 1)).to(points.device)
    affine_nodes0 = torch.cat((points, z), 3)
    affine_nodes0 = affine_nodes0.squeeze(1)
    affine_nodes0 = torch.bmm(affine_nodes0, affine_pars)
    affine_nodes0 = affine_nodes0.unsqueeze(1)
    affine_nodes0 = affine_nodes0[:, :, :, :2]
    return affine_nodes0


def similarity_matrix(affine_pars1, affine_pars2):
    """

    Args:
        affine_pars1: (B, 6), rotation rx, ry, rz, translation x, y, z
        affine_pars2: (B, 3), scale x, y, z

    Returns:
        Affine_matrix: (B, 4, 4)

    """
    theta_z = affine_pars1[:, 2] * math.pi
    rotation_matrix_z = torch.stack([
        torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_z), torch.zeros_like(theta_z)], dim=1),
        torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_z), torch.zeros_like(theta_z)], dim=1),
        torch.stack([torch.zeros_like(theta_z), torch.zeros_like(theta_z), torch.ones_like(theta_z), torch.zeros_like(theta_z)], dim=1),
        torch.stack([torch.zeros_like(theta_z), torch.zeros_like(theta_z), torch.zeros_like(theta_z), torch.ones_like(theta_z)],dim=1)
    ], dim=2)

    theta_x = affine_pars1[:, 0] * math.pi
    rotation_matrix_x = torch.stack([
        torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)],dim=1),
        torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), torch.sin(theta_x), torch.zeros_like(theta_x)], dim=1),
        torch.stack([torch.zeros_like(theta_x), -torch.sin(theta_x), torch.cos(theta_x), torch.zeros_like(theta_x)], dim=1),
        torch.stack([torch.zeros_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x), torch.ones_like(theta_x), ], dim=1),
    ], dim=2)

    theta_y = affine_pars1[:, 1] * math.pi
    rotation_matrix_y = torch.stack([
        torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), -torch.sin(theta_y), torch.zeros_like(theta_y)],dim=1),
        torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_y), torch.zeros_like(theta_y), torch.zeros_like(theta_y)],dim=1),
        torch.stack([torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y), torch.zeros_like(theta_y)], dim=1),
        torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_y), ], dim=1),
    ], dim=2)
    cx = affine_pars2[:, 0]
    cy = affine_pars2[:, 1]
    cz = affine_pars2[:, 2]
    scaling_matrix = torch.stack([
        torch.stack([cx, torch.zeros_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=1),
        torch.stack([torch.zeros_like(cx), cy, torch.zeros_like(cx), torch.zeros_like(cx)], dim=1),
        torch.stack([torch.zeros_like(cx), torch.zeros_like(cx), cz, torch.zeros_like(cx)], dim=1),
        torch.stack([torch.zeros_like(cx), torch.zeros_like(cx), torch.zeros_like(cx), torch.ones_like(cx)], dim=1)
    ], dim=2)

    tx = affine_pars1[:, 3]
    ty = affine_pars1[:, 4]
    tz = affine_pars1[:, 5]

    translation_matrix = torch.stack([
        torch.stack([torch.ones_like(tx), torch.zeros_like(tx), torch.zeros_like(tx), tx], dim=1),
        torch.stack([torch.zeros_like(tx), torch.ones_like(tx), torch.zeros_like(tx), ty], dim=1),
        torch.stack([torch.zeros_like(tx), torch.zeros_like(tx), torch.ones_like(tx), tz], dim=1),
        torch.stack([torch.zeros_like(tx), torch.zeros_like(tx), torch.zeros_like(tx), torch.ones_like(tx)], dim=1)
    ], dim=2)
    rotation_matrix = torch.bmm(torch.bmm(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    affine_matrix = torch.bmm(torch.bmm(rotation_matrix, scaling_matrix), translation_matrix)
    return affine_matrix


class CardiacUNet(torch.nn.Module):
    def __init__(self, enc_dim, drop, output_dim):
        super().__init__()
        self.backbone = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=1)
        self.decoder = Decoder(
            self.backbone.dims, drop=drop, kernel_size=kernel_size, output_act="sigmoid", output_dim=output_dim
        )

    def forward(self, img):
        # x = (B, 1, H, W)
        x = norm_tensor(img)
        features = self.backbone(x)
        output = self.decoder(features)
        return output


def plot(voxel_map):
    from mayavi import mlab
    voxel_map = voxel_map.squeeze().detach().cpu().numpy()
    xx, yy, zz = np.where(voxel_map[0, ...] == 1)

    cube = mlab.points3d(xx, yy, zz,
                 mode="cube",
                 color=(0, 1, 0),
                 scale_factor=1)
    mlab.outline()

    mlab.show()
    # from matplotlib import pyplot as plt
    # half_dim = 128/2
    # nodes0[:, :, :, 1] = -nodes0[:, :, :, 1]
    # nodes0 = nodes0 * half_dim + half_dim
    #
    # nodes1[:, :, :, 1] = -nodes1[:, :, :, 1]
    # nodes1 = nodes1 * half_dim + half_dim
    #
    # nodes2[:, :, :, 1] = -nodes2[:, :, :, 1]
    # nodes2 = nodes2 * half_dim + half_dim
    #
    # nodes0 = nodes0[0, 0, :, :].detach().cpu().numpy()
    # nodes1 = nodes1[0, 0, :, :].detach().cpu().numpy()
    # nodes2 = nodes2[0, 0, :, :].detach().cpu().numpy()
    #
    # face0 = face0[0, 0, :, :].detach().cpu().numpy()
    # face1 = face1[0, 0, :, :].detach().cpu().numpy()
    # face2 = face2[0, 0, :, :].detach().cpu().numpy()
    # # plt.show()
    # # plt.figure()
    # plt.imshow(np.zeros((128, 128)))
    # plt.plot(nodes2[0, 0], nodes2[0, 1], 'go')
    # plt.plot(nodes2[-1, 0], nodes2[-1, 1], 'ro')
    #
    # plt.plot(nodes2[:, 0], nodes2[:, 1], 'bx-')
    # for f in range(face2.shape[0]):
    #     p1 = nodes2[face2[f, 0], :]
    #     p2 = nodes2[face2[f, 1], :]
    #     p3 = nodes2[face2[f, 2], :]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    #     plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
    #     plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')
    #
    # plt.figure()
    # plt.imshow(np.zeros((128, 128)))
    # plt.plot(nodes0[0, 0], nodes0[0, 1], 'go')
    # plt.plot(nodes0[-1, 0], nodes0[-1, 1], 'ro')
    #
    # plt.plot(nodes0[:, 0], nodes0[:, 1], 'bx-')
    # for f in range(face0.shape[0]):
    #     p1 = nodes0[face0[f, 0], :]
    #     p2 = nodes0[face0[f, 1], :]
    #     p3 = nodes0[face0[f, 2], :]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    #     plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
    #     plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')
    #
    # plt.figure()
    # plt.imshow(np.zeros((128, 128)))
    # plt.plot(nodes1[0, 0], nodes1[0, 1], 'go')
    # plt.plot(nodes1[-1, 0], nodes1[-1, 1], 'ro')
    #
    # plt.plot(nodes1[:, 0], nodes1[:, 1], 'bx-')
    # for f in range(face1.shape[0]):
    #     p1 = nodes1[face1[f, 0], :]
    #     p2 = nodes1[face1[f, 1], :]
    #     p3 = nodes1[face1[f, 2], :]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    #     plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
    #     plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')
    #
    # plt.show()


def elastic_v(nodes):
    """nodes: (B, 1, N, 2)"""
    Pf = nodes.roll(-1, dims=2)
    Pb = nodes.roll(1, dims=2)

    K = Pf + Pb - 2 * nodes
    return K


def stiff_v(nodes):
    """nodes: (B, 1, N, 2)"""
    Pf = nodes.roll(-1, dims=2)
    Pff = Pf.roll(-1, dims=2)

    Pb = nodes.roll(1, dims=2)
    Pbb = Pb.roll(1, dims=2)

    K = - Pff + Pf * 4 - 6 * nodes - Pb * 4 - Pbb * 2
    return K
