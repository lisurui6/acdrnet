import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from backbones.unet import Encoder, Decoder
from utils.topology import get_circles, get_circles_2, get_circles_3
import neural_renderer as nr
from torch.distributions.normal import Normal
import math
from models.layer import SpatialTransformer, VecInt, AffineSpatialTransformer


__all__ = ['ShapeDeformCardiacCircleNet']

kernel_size = 5


class ShapeDeformCardiacCircleNet(nn.Module):
    """No points, just mask after init. then affine and deform"""
    def __init__(self, num_nodes, enc_dim, dec_size, image_size, drop):
        super().__init__()
        self.num_nodes = num_nodes
        self.dec_size = dec_size
        self.image_size = image_size

        padding = int((kernel_size - 1) / 2)

        self.shape_encoder = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=1)
        self.shape_regressor = nn.Sequential(
            nn.Conv2d(self.shape_encoder.dims[-2], self.shape_encoder.dims[-2] // 2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(self.shape_encoder.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(self.shape_encoder.dims[-2] // 2, self.shape_encoder.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.shape_encoder.dims[-2] // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )

        self.shape_end1 = nn.Sequential(nn.Linear(200, 2), nn.Tanh())  # c0(x,y)
        # bias = torch.from_numpy(np.array([0, 0, 0, 33/128])).float()
        bias = torch.from_numpy(np.array([0, 0])).float()

        self.shape_end1[0].weight.data.zero_()
        self.shape_end1[0].bias.data.copy_(bias)

        self.shape_end2 = nn.Sequential(nn.Linear(200, 5), nn.Sigmoid())  # r1, factor 0 (r0/r1), theta2/pi, d_c2_c0, theta_c2
        # bias = torch.from_numpy(np.array([-1.1838, 0.690, -1.6094, 0, -1.099])).float()
        bias = torch.from_numpy(np.array([-1.1838, 0.690, -1.6094, 0, 1.9459])).float()

        # self.shape_end2 = nn.Sequential(nn.Linear(200, 3), nn.Sigmoid())  # factor 0 (r0/r1), theta2/pi, d_c2_c0
        # bias = torch.from_numpy(np.array([0.690, -1.6094, 0])).float()
        self.shape_end2[0].weight.data.zero_()
        self.shape_end2[0].bias.data.copy_(bias)

        self.affine_encoder = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=4)
        self.affine_regressor = nn.Sequential(
            nn.Conv2d(self.affine_encoder.dims[-2], self.affine_encoder.dims[-2] // 2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(self.affine_encoder.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(self.affine_encoder.dims[-2] // 2, self.affine_encoder.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.affine_encoder.dims[-2] // 4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        self.affine_end1 = nn.Sequential(nn.Linear(200, 3), nn.Tanh())  # rotation, translation x, y,
        bias = torch.from_numpy(np.array([0, 0, 0])).float()
        self.affine_end1[0].weight.data.zero_()
        self.affine_end1[0].bias.data.copy_(bias)

        self.affine_end2 = nn.Sequential(nn.Linear(200, 2))  # scale x, y
        bias = torch.from_numpy(np.array([1, 1])).float()
        self.affine_end2[0].weight.data.zero_()
        self.affine_end2[0].bias.data.copy_(bias)

        self.deform_backbone = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=4)
        self.decoder = Decoder(
            self.deform_backbone.dims, drop=drop, kernel_size=kernel_size, output_act=None, output_dim=self.deform_backbone.dims[-4]
        )

        self.flow = nn.Conv2d(self.deform_backbone.dims[-4], 2, kernel_size=3, padding=1)
        # # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.texture_size = 2
        self.camera_distance = 1
        self.elevation = 0
        self.azimuth = 0
        self.renderer = nr.Renderer(
            camera_mode='look_at',
            image_size=self.image_size,
            light_intensity_ambient=1,
            light_intensity_directional=1,
            perspective=False
        )
        self.integrate = VecInt(
            inshape=(image_size, image_size),
            nsteps=7,
        )
        self.tri0 = None
        self.tri1 = None
        self.tri2 = None
        self.affine_transformer = AffineSpatialTransformer(mode="bilinear")
        self.deform_transformer = SpatialTransformer(size=(image_size, image_size), mode="bilinear")

    def forward(self, img, steps=7):
        # x = (B, 1, H, W)
        x = norm_tensor(img)
        out = self.shape_encoder(x)
        out = self.shape_regressor(out[-1])
        out_par1 = self.shape_end1(out)
        out_par2 = self.shape_end2(out)

        nodes0, faces0, nodes1, faces1, nodes2, faces2, self.tri0, self.tri1, self.tri2 = get_circles_3(
            out_par1, out_par2,
            img.shape[0], self.dec_size, self.num_nodes, img.device, self.tri0, self.tri1, self.tri2
        )

        # plot(nodes0, faces0, nodes1, faces1, nodes2, faces2)
        init_mask0 = self.render_mask(nodes0, faces0)
        init_mask1 = self.render_mask(nodes1, faces1)
        init_mask2 = self.render_mask(nodes2, faces2)

        affine_in = torch.cat([x, init_mask0, init_mask1, init_mask2], dim=1).detach()
        out = self.affine_encoder(affine_in)
        affine_pars = self.affine_regressor(out[-1])

        affine_pars1 = self.affine_end1(affine_pars)
        affine_pars2 = self.affine_end2(affine_pars)

        #
        affine_matrix = similarity_matrix(affine_pars1, affine_pars2)
        affine_theta = affine_matrix[:, :2, :]
        init_mask = torch.cat([init_mask0, init_mask1, init_mask2], dim=1)
        affine_mask = self.affine_transformer(init_mask, affine_theta)

        deform_in = torch.cat([x, affine_mask], dim=1).detach()
        features = self.deform_backbone(deform_in)
        flow = self.flow(self.decoder(features))
        #
        if flow.shape[2:] != img.shape[2:]:
            flow = F.interpolate(flow, size=img.shape[2:], mode='bilinear')
        preint_flow = flow
        flow = self.integrate(preint_flow)
        deform_mask = self.deform_transformer(affine_mask, flow)

        if self.training:
            return flow, preint_flow, init_mask0, init_mask1, init_mask2, \
                   affine_mask[:, 0:1, ...], affine_mask[:, 1:2, ...], affine_mask[:, 2:, ...], \
                   deform_mask[:, 0:1, ...], deform_mask[:, 1:2, ...], deform_mask[:, 2:, ...]
        else:
            return flow, init_mask0, init_mask1, init_mask2, \
                   affine_mask[:, 0:1, ...], affine_mask[:, 1:2, ...], affine_mask[:, 2:, ...], \
                   deform_mask[:, 0:1, ...], deform_mask[:, 1:2, ...], deform_mask[:, 2:, ...]

    def render_mask(self, nodes, faces):
        z = torch.ones((nodes.shape[0], 1, nodes.shape[2], 1)).to(nodes.device)
        P3d = torch.cat((nodes, z), 3)
        P3d = torch.squeeze(P3d, dim=1)
        faces = torch.squeeze(faces, dim=1).to(nodes.device)
        mask = self.renderer(P3d, faces, mode='silhouettes').unsqueeze(1)
        return mask


def norm_tensor(_x):
    bs = _x.shape[0]
    min_x = _x.view(bs, 1, -1).min(dim=2)[0].view(bs, 1, 1, 1)
    max_x = _x.view(bs, 1, -1).max(dim=2)[0].view(bs, 1, 1, 1)
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
    theta = affine_pars1[:, 0] * math.pi
    rotation_matrix = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)], dim=1),
        torch.stack([torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)], dim=1),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=1)
    ], dim=2)
    cx = affine_pars2[:, 0]
    cy = affine_pars2[:, 1]
    scaling_matrix = torch.stack([
        torch.stack([cx, torch.zeros_like(theta), torch.zeros_like(theta)], dim=1),
        torch.stack([torch.zeros_like(theta), cy, torch.zeros_like(theta)], dim=1),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=1)
    ], dim=2)

    tx = affine_pars1[:, 1]
    ty = affine_pars1[:, 2]

    translation_matrix = torch.stack([
        torch.stack([torch.ones_like(theta), torch.zeros_like(theta), tx], dim=1),
        torch.stack([torch.zeros_like(theta), torch.ones_like(theta), ty], dim=1),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=1)
    ], dim=2)

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


def plot(nodes0, face0, nodes1, face1, nodes2, face2):
    from matplotlib import pyplot as plt
    half_dim = 128/2
    nodes0[:, :, :, 1] = -nodes0[:, :, :, 1]
    nodes0 = nodes0 * half_dim + half_dim

    nodes1[:, :, :, 1] = -nodes1[:, :, :, 1]
    nodes1 = nodes1 * half_dim + half_dim

    nodes2[:, :, :, 1] = -nodes2[:, :, :, 1]
    nodes2 = nodes2 * half_dim + half_dim

    nodes0 = nodes0[0, 0, :, :].detach().cpu().numpy()
    nodes1 = nodes1[0, 0, :, :].detach().cpu().numpy()
    nodes2 = nodes2[0, 0, :, :].detach().cpu().numpy()

    face0 = face0[0, 0, :, :].detach().cpu().numpy()
    face1 = face1[0, 0, :, :].detach().cpu().numpy()
    face2 = face2[0, 0, :, :].detach().cpu().numpy()
    # plt.show()
    # plt.figure()
    plt.imshow(np.zeros((128, 128)))
    plt.plot(nodes2[0, 0], nodes2[0, 1], 'go')
    plt.plot(nodes2[-1, 0], nodes2[-1, 1], 'ro')

    plt.plot(nodes2[:, 0], nodes2[:, 1], 'bx-')
    for f in range(face2.shape[0]):
        p1 = nodes2[face2[f, 0], :]
        p2 = nodes2[face2[f, 1], :]
        p3 = nodes2[face2[f, 2], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
        plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

    plt.figure()
    plt.imshow(np.zeros((128, 128)))
    plt.plot(nodes0[0, 0], nodes0[0, 1], 'go')
    plt.plot(nodes0[-1, 0], nodes0[-1, 1], 'ro')

    plt.plot(nodes0[:, 0], nodes0[:, 1], 'bx-')
    for f in range(face0.shape[0]):
        p1 = nodes0[face0[f, 0], :]
        p2 = nodes0[face0[f, 1], :]
        p3 = nodes0[face0[f, 2], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
        plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

    plt.figure()
    plt.imshow(np.zeros((128, 128)))
    plt.plot(nodes1[0, 0], nodes1[0, 1], 'go')
    plt.plot(nodes1[-1, 0], nodes1[-1, 1], 'ro')

    plt.plot(nodes1[:, 0], nodes1[:, 1], 'bx-')
    for f in range(face1.shape[0]):
        p1 = nodes1[face1[f, 0], :]
        p2 = nodes1[face1[f, 1], :]
        p3 = nodes1[face1[f, 2], :]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
        plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

    plt.show()


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
