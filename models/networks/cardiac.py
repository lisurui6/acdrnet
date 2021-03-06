import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from backbones.unet import Encoder, Decoder
from utils.topology import get_circles
import neural_renderer as nr

__all__ = ['CardaicCircleNet']

kernel_size = 5


class CardaicCircleNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, dec_size, image_size, drop):
        super().__init__()
        self.num_nodes = num_nodes
        self.dec_size = dec_size
        self.image_size = image_size

        self.backbone = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=1)
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
            nn.Linear(200, 6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0])).float()
        self.affine_regressor[-2].weight.data.zero_()
        self.affine_regressor[-2].bias.data.copy_(bias)
        self.disp = Decoder(
            self.backbone.dims, drop=drop, kernel_size=kernel_size, output_act=None, output_dim=6
        )

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

    def forward(self, img, iter=3):
        # x = (B, 1, H, W)
        nodes0, faces0, nodes1, faces1, nodes2, faces2, cp0, cp1 = get_circles(img.shape[0], self.dec_size, self.num_nodes, img.device)
        plot(nodes0, faces0, nodes1, faces1, nodes2, faces2)
        x = norm_tensor(img)
        features = self.backbone(x)
        affine_pars = self.affine_regressor(features[-1])
        affine_pars = affine_pars.view(affine_pars.shape[0], 2, 3)
        affine_node0 = affine_transform_points(nodes0, affine_pars)
        affine_node1 = affine_transform_points(nodes1, affine_pars)
        affine_node2 = affine_transform_points(nodes2, affine_pars)

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

        disp = torch.tanh(self.disp(features))

        if disp.shape[2:] != img.shape[2:]:
            disp = F.interpolate(disp, size=img.shape[2:], mode='bilinear')
        for i in range(iter):
            # Sample and move
            affine_node0[..., 1] = affine_node0[..., 1] * -1
            Pxx = F.grid_sample(disp[:, 0:1], affine_node0).transpose(3, 2)
            Pyy = F.grid_sample(disp[:, 1:2], affine_node0).transpose(3, 2)
            dP0 = torch.cat((Pxx, Pyy), -1)
            affine_node0 = affine_node0 + dP0

            affine_node1[..., 1] = affine_node1[..., 1] * -1
            Pxx = F.grid_sample(disp[:, 2:3], affine_node1).transpose(3, 2)
            Pyy = F.grid_sample(disp[:, 3:4], affine_node1).transpose(3, 2)
            dP1 = torch.cat((Pxx, Pyy), -1)
            affine_node1 = affine_node1 + dP1

            affine_node2[..., 1] = affine_node2[..., 1] * -1
            Pxx2 = F.grid_sample(disp[:, 4:5], affine_node2).transpose(3, 2)
            Pyy2 = F.grid_sample(disp[:, 5:6], affine_node2).transpose(3, 2)
            dP2 = torch.cat((Pxx2, Pyy2), -1)
            nodes = []
            for i in range(affine_node2.shape[2]):
                if i < cp0:
                    node = affine_node2[:, :, i] + dP2[:, :, i, :]
                if i == cp0:
                    node = affine_node1[:, :, self.num_nodes//3].clone() + dP2[:, :, i, :]
                if i > cp0:
                    node = affine_node1[:, :, self.num_nodes//3 - (i - cp0)].clone()
                nodes.append(node)
            nodes[-1] += dP2[:, :, -1, :]
            affine_node2 = torch.stack(nodes, dim=2)

            # Render mask
            affine_node0[..., 1] = affine_node0[..., 1] * -1
            affine_node1[..., 1] = affine_node1[..., 1] * -1
            affine_node2[..., 1] = affine_node2[..., 1] * -1

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
            return output_masks0, output_masks1, output_masks2, output_points0, output_points1, output_points2, disp
        else:
            return output_masks0[0], output_masks0[-1], output_masks1[0], output_masks1[-1], output_masks2[0], output_masks2[-1], disp

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
