import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from backbones.unet import Encoder, Decoder
from utils.topology import get_circles, get_circles_2
import neural_renderer as nr
from torch.distributions.normal import Normal
import math
from models.layer import SpatialTransformer, VecInt


__all__ = ['DeformCardiacCircleNet']

kernel_size = 5


class DeformCardiacCircleNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, dec_size, image_size, drop):
        super().__init__()
        self.num_nodes = num_nodes
        self.dec_size = dec_size
        self.image_size = image_size

        self.backbone = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=4)
        padding = int((kernel_size - 1) / 2)

        self.affine_regressor = nn.Sequential(
            nn.Conv2d(self.backbone.dims[-2], self.backbone.dims[-2] // 2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(self.backbone.dims[-2] // 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(self.backbone.dims[-2] // 2, self.backbone.dims[-2] // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.backbone.dims[-2] // 4),
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

        self.decoder = Decoder(
            self.backbone.dims, drop=drop, kernel_size=kernel_size, output_act=None, output_dim=self.backbone.dims[-4]
        )

        self.flow = nn.Conv2d(self.backbone.dims[-4], 2, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
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

    def forward(self, img, steps=7):
        # x = (B, 1, H, W)
        nodes0, faces0, nodes1, faces1, nodes2, faces2, cp0, cp1 = get_circles_2(img.shape[0], self.dec_size, self.num_nodes, img.device)

        # plot(nodes0, faces0, nodes1, faces1, nodes2, faces2)
        x = norm_tensor(img)
        init_mask0 = self.render_mask(nodes0, faces0)
        init_mask1 = self.render_mask(nodes1, faces1)
        init_mask2 = self.render_mask(nodes2, faces2)
        x = torch.cat([x, init_mask0, init_mask1, init_mask2], dim=1)
        features = self.backbone(x)
        affine_pars = self.affine_regressor(features[-1])

        affine_pars1 = self.affine_end1(affine_pars)
        affine_pars2 = self.affine_end2(affine_pars)

        # affine_pars = affine_pars.view(affine_pars.shape[0], 2, 3)
        # affine_node0 = affine_transform_points(nodes0, affine_pars)
        # affine_node1 = affine_transform_points(nodes1, affine_pars)
        # affine_node2 = affine_transform_points(nodes2, affine_pars)

        affine_node0 = similarity_transform_points(nodes0, affine_pars1, affine_pars2)
        affine_node1 = similarity_transform_points(nodes1, affine_pars1, affine_pars2)
        affine_node2 = similarity_transform_points(nodes2, affine_pars1, affine_pars2)

        affine_mask0 = self.render_mask(affine_node0, faces0)
        affine_mask1 = self.render_mask(affine_node1, faces1)
        affine_mask2 = self.render_mask(affine_node2, faces2)

        #
        output_masks0 = [affine_mask0]
        output_masks1 = [affine_mask1]
        output_masks2 = [affine_mask2]

        output_points0 = []
        output_points1 = []
        output_points2 = []

        flow = self.flow(self.decoder(features))

        if flow.shape[2:] != img.shape[2:]:
            flow = F.interpolate(flow, size=img.shape[2:], mode='bilinear')
        flow = flow / img.shape[2]
        scale = 1.0 / steps
        flow = flow * scale
        for i in range(steps):
            # Sample and move
            affine_node0[..., 1] = affine_node0[..., 1] * -1
            Pxx = F.grid_sample(flow[:, 0:1], affine_node0).transpose(3, 2)
            Pyy = F.grid_sample(flow[:, 1:2], affine_node0).transpose(3, 2)
            dP0 = torch.cat((Pxx, Pyy), -1)
            affine_node0 = affine_node0 + dP0

            affine_node1[..., 1] = affine_node1[..., 1] * -1
            Pxx = F.grid_sample(flow[:, 0:1], affine_node1).transpose(3, 2)
            Pyy = F.grid_sample(flow[:, 1:2], affine_node1).transpose(3, 2)
            dP1 = torch.cat((Pxx, Pyy), -1)
            affine_node1 = affine_node1 + dP1

            affine_node2[..., 1] = affine_node2[..., 1] * -1
            Pxx2 = F.grid_sample(flow[:, 0:1], affine_node2).transpose(3, 2)
            Pyy2 = F.grid_sample(flow[:, 1:2], affine_node2).transpose(3, 2)
            dP2 = torch.cat((Pxx2, Pyy2), -1)
            nodes = []
            for i in range(affine_node2.shape[2]):
                if i < cp0:
                    node = affine_node2[:, :, i] + dP2[:, :, i, :]
                if i == cp0:
                    node = affine_node1[:, :, self.num_nodes//3].clone()
                if i > cp0:
                    node = affine_node1[:, :, self.num_nodes//3 - (i - cp0)].clone()
                nodes.append(node)
            affine_node2 = torch.stack(nodes, dim=2)

            # Render mask
            affine_node0[..., 1] = affine_node0[..., 1] * -1
            affine_node1[..., 1] = affine_node1[..., 1] * -1
            affine_node2[..., 1] = affine_node2[..., 1] * -1

            # compute internal v
            # dP0 = elastic_v(affine_node0)
            # dP1 = elastic_v(affine_node1)
            # dP2 = elastic_v(affine_node1)
            # affine_node0 = affine_node0 + dP0
            # affine_node1 = affine_node1 + dP1
            # affine_node2[:, :, :cp0] = affine_node2[:, :, :cp0] + dP2[:, :, :cp0]

            # dP0 = stiff_v(affine_node0)
            # dP1 = stiff_v(affine_node1)
            # dP2 = stiff_v(affine_node1)
            # affine_node0 = affine_node0 + dP0
            # affine_node1 = affine_node1 + dP1
            # affine_node2[:, :, :cp0] = affine_node2[:, :, :cp0] + dP2[:, :, :cp0]

        # Render mask
        mask0 = self.render_mask(affine_node0, faces0)
        mask1 = self.render_mask(affine_node1, faces1)
        mask2 = self.render_mask(affine_node2, faces2)

        output_masks0.append(mask0)
        output_masks1.append(mask1)
        output_masks2.append(mask2)

        output_points0.append(affine_node0)
        output_points1.append(affine_node1)
        output_points2.append(affine_node2)

        if self.training:
            return output_masks0, output_masks1, output_masks2, output_points0, output_points1, output_points2, flow
        else:
            return output_masks0[0], output_masks0[-1], output_masks1[0], output_masks1[-1], output_masks2[0], output_masks2[-1], flow

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


def similarity_transform_points(points, affine_pars1, affine_pars2):
    z = torch.ones((points.shape[0], 1, points.shape[2], 1)).to(points.device)
    affine_nodes0 = torch.cat((points, z), 3)
    affine_nodes0 = affine_nodes0.squeeze(1)

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
    affine_nodes0 = torch.bmm(affine_nodes0, affine_matrix)

    affine_nodes0 = affine_nodes0.unsqueeze(1)
    affine_nodes0 = affine_nodes0[:, :, :, :2]
    return affine_nodes0


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
