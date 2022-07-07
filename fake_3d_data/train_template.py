import torch
import numpy as np
from rasterizor.voxelize import Voxelize
import nibabel as nib
from typing import Tuple
from scipy.ndimage import zoom
from torch.autograd import Variable
from matplotlib import pyplot as plt
from mayavi import mlab
from utils.topology_3d_smooth_lv import sample_3d_points
from torch.distributions.normal import Normal
from models.layer import SpatialTransformer, VecInt
from backbones.unet_3d import Encoder, Decoder
import torch.nn.functional as F
from models.loss_functions import Grad3D
import torch.nn as nn
from fake_3d_data.sample import sample_lv_points

voxel_width = 32
voxel_depth = 32
voxel_height = 32

kernel_size = 5


def norm_tensor(_x):
    bs = _x.shape[0]
    min_x = _x.view(bs, 1, -1).min(dim=2)[0].view(bs, 1, 1, 1, 1)
    max_x = _x.view(bs, 1, -1).max(dim=2)[0].view(bs, 1, 1, 1, 1)
    return (_x - min_x) / (max_x - min_x + 1e-2)


class LVShapeNet(torch.nn.Module):
    def __init__(self, voxel_width, voxel_depth, voxel_height, num_lv_slices):
        super().__init__()
        self.voxel_width = voxel_width
        self.voxel_height = voxel_height
        self.voxel_depth = voxel_depth
        self.num_lv_slices = num_lv_slices

        padding = int((kernel_size - 1) / 2)

        self.shape_encoder = Encoder(512, drop=False, kernel_size=kernel_size, in_channels=1)
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
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )

        self.shape_end1 = nn.Sequential(nn.Linear(200, 4), nn.Tanh())  # c0_x, c0_y, c0_z, c0_z_end
        bias = torch.from_numpy(np.array([0, 0, -0.5, 0.5])).float()
        self.shape_end1[0].weight.data.zero_()
        self.shape_end1[0].bias.data.copy_(bias)

        self.shape_end2 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())  # r0
        bias = torch.from_numpy(np.array([-1.7])).float()

        self.shape_end2[0].weight.data.zero_()
        self.shape_end2[0].bias.data.copy_(bias)

        num_tanh_delta = (num_lv_slices - 1) * 3  # dx, dy, dr: dx = [-1, 1] * d_max
        self.shape_end3 = nn.Sequential(nn.Linear(200, num_tanh_delta), nn.Tanh())
        d_bias = np.random.normal(0, 1, num_tanh_delta)
        d_bias[(num_lv_slices - 1) * 2:] = np.abs(d_bias[(num_lv_slices - 1) * 2:])
        bias = torch.from_numpy(d_bias).float()
        self.shape_end3[0].weight.data.zero_()
        self.shape_end3[0].bias.data.copy_(bias)
        self.tri0 = None
        self.voxeliser = Voxelize(voxel_width=self.voxel_width, voxel_height=self.voxel_height, voxel_depth=self.voxel_depth, eps=1e-4, eps_in=20)

    def forward(self, img, epoch=0, vis=False):
        # x = (B, 1, H, W)
        x = norm_tensor(img)
        out = self.shape_encoder(x)
        out = self.shape_regressor(out[-1])
        out_par1 = self.shape_end1(out)  # (B, 3), tanh
        out_par2 = self.shape_end2(out)  # (B, 5), sig
        out_par3 = self.shape_end3(out)  # (B, 5), sig

        nodes0, tetras0, self.tri0 = sample_lv_points(
            par1=out_par1, par2=out_par2, par3=out_par3, num_lv_slices=self.num_lv_slices,
            voxel_depth=self.voxel_depth, voxel_height=self.voxel_height, voxel_width=self.voxel_width,
            num_points=32, batch_size=img.shape[0], lv_tetras=self.tri0, epoch=epoch,
        )
        init_mask0 = self.voxelize_mask(nodes0, tetras0)

        if self.training:
            return init_mask0, nodes0
        else:
            return init_mask0

    def voxelize_mask(self, nodes, faces):
        P3d = torch.squeeze(nodes, dim=1)
        faces = torch.squeeze(faces, dim=1).to(nodes.device)
        mask = self.voxeliser(P3d, faces).unsqueeze(1)
        return mask


class DeformNet(torch.nn.Module):
    """Deform a ball to match template"""
    def __init__(self, enc_dim, voxel_width, voxel_height, voxel_depth, drop):
        super().__init__()
        self.voxel_width = voxel_width
        self.voxel_height = voxel_height
        self.voxel_depth = voxel_depth

        self.deform_backbone = Encoder(enc_dim, drop=drop, kernel_size=kernel_size, in_channels=2)
        self.decoder = Decoder(
            self.deform_backbone.dims, drop=drop, kernel_size=kernel_size, output_act=None,
            output_dim=self.deform_backbone.dims[-4]
        )

        self.flow_conv = torch.nn.Conv3d(self.deform_backbone.dims[-4], 3, kernel_size=3, padding=1)
        # # init flow layer with small weights and bias
        self.flow_conv.weight = torch.nn.Parameter(Normal(0, 1e-5).sample(self.flow_conv.weight.shape))
        self.flow_conv.bias = torch.nn.Parameter(torch.zeros(self.flow_conv.bias.shape))

        self.integrate = VecInt(
            inshape=(self.voxel_width, self.voxel_height, self.voxel_depth),
            nsteps=7,
        )

        self.deform_transformer = SpatialTransformer(size=(self.voxel_width, self.voxel_height, self.voxel_depth), mode="bilinear")
        self.voxeliser = Voxelize(voxel_width=self.voxel_width, voxel_height=self.voxel_height, voxel_depth=self.voxel_depth, eps=1e-4, eps_in=20)

    def forward(self, vertices, facets, image):
        # x = (B, 1, H, W)
        x = norm_tensor(image)

        init_mask = self.voxelize_mask(vertices, facets)

        deform_in = torch.cat([x, init_mask], dim=1).detach()
        features = self.deform_backbone(deform_in)
        flow = self.flow_conv(self.decoder(features))  # (B, 3, W, H, D) (dx, dy, dz)
        flow = flow / image.shape[2]

        preint_flow = flow
        flow = self.integrate(preint_flow)
        # vertices: (B, D, 3)
        vertices[..., 1] = vertices[..., 1] * -1
        Pxx = F.grid_sample(flow[:, 0:1], vertices.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)  # Pxx (B, D, 1)
        Pyy = F.grid_sample(flow[:, 1:2], vertices.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)
        Pzz = F.grid_sample(flow[:, 2:3], vertices.unsqueeze(2).unsqueeze(2)).squeeze(-1).squeeze(-1).transpose(1, 2)

        dP0 = torch.cat((Pxx, Pyy, Pzz), -1)
        vertices = vertices + dP0

        deform_mask = self.voxelize_mask(vertices, facets)

        return flow, preint_flow, init_mask, deform_mask, vertices

    def voxelize_mask(self, nodes, faces):
        P3d = torch.squeeze(nodes, dim=1)
        faces = torch.squeeze(faces, dim=1).to(nodes.device)
        mask = self.voxeliser(P3d, faces).unsqueeze(1)
        return mask


def resize_image(image: np.ndarray, target_shape: Tuple, order: int):
    image_shape = image.shape
    factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
    output = zoom(image, factors, order=order)
    return output


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def is_on_the_same_plane(v0, v1, v2, v3):
    n = np.cross(v1 - v0, v2-v0)
    n /= np.linalg.norm(n)
    return np.dot(v3 - v0, n)


image = nib.load("image.nii.gz").get_data()
if image.ndim == 4:
    image = np.squeeze(image, axis=-1).astype(np.int16)
image = image.astype(np.float32)
image = resize_image(image, (voxel_width, voxel_depth, voxel_height), 0)
# image = np.transpose(image, (2, 0, 1))
image = rescale_intensity(image, (1.0, 99.0))
image = np.expand_dims(image, 0)
image = torch.from_numpy(image).float()  # (1, w, h, d)


label = nib.load("seg_lvsa_SR_ED.nii.gz").get_data()
if label.ndim == 4:
    label = np.squeeze(label, axis=-1).astype(np.int16)
label = label.astype(np.float32)
label[label == 4] = 3
label = resize_image(label, (voxel_width, voxel_depth, voxel_height), 0)

lv_label = np.zeros_like(label)
lv_label[[label == 1]] = 1  # 1: LV endo, 2: LV myo, 3,4: RV
lv_myo_label = np.zeros_like(label)
lv_myo_label[[label == 2]] = 1
rv_label = np.zeros_like(label)
rv_label[[label == 3]] = 1

# image = np.transpose(image, (2, 0, 1))

label = np.stack([lv_label, lv_myo_label, rv_label], axis=0)
label = torch.from_numpy(label).float().cuda()
xx, yy, zz = np.where(label[0].detach().cpu().numpy() >= 0)

cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 0, 0),
                     scale_factor=1, transparent=True, opacity=0)
mlab.outline()

xx, yy, zz = np.where(label[0].detach().cpu().numpy() >= 0.5)
cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 1, 0),
                     scale_factor=1, transparent=True, opacity=1)
mlab.outline()
mlab.show()

# par1 = (np.array([[16, 16, 16]]) - voxel_height/2) / voxel_height * 2
# par2 = np.array([[5]]) / voxel_height * 2

# par1 = torch.from_numpy(par1).float().cuda()
# par2 = torch.from_numpy(par2).float().cuda()
# vertices, facets, _ = sample_3d_points(
#     par1, par2, None, voxel_width, voxel_height, voxel_depth,
#     10, 1, None,
# )

# model = DeformNet(512, voxel_width, voxel_height, voxel_depth, 0)
# model.cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# print(facets, facets.shape)
# losses = []
# flow_grad_loss = Grad3D(penalty="l2")
# from tqdm import tqdm
# pbar = tqdm(range(1000000))
# for i in pbar:
#     # print(vertices.shape, facets.shape)
#     flow, preint_flow, init_mask, deform_mask, output_vertices = model(vertices, facets, label)
#
#     # mse_loss = loss(pred, label)
#     # print(mse_loss.item())
#     # mse_loss.backward()
#     loss = (label - deform_mask).pow(2).mean()
#     loss += flow_grad_loss(flow)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # print("loss", i, loss.item())
#     # print((vertices[0] * 64 + 63) / 2)
#     # print(i, vertices.grad[0])
#     losses.append(loss.item())
#     torch.save(model.state_dict(), "CP.pth")
#     pbar.set_description("Loss: {}".format(loss.item()))
#     # np_vertices = ((output_vertices[0] * voxel_depth + voxel_height - 1) / 2).detach().cpu().numpy()
#     if i % 1000 == 0:
#         plot = True
#     else:
#         plot = False
#     if plot:
#         map = deform_mask[0, 0].detach().cpu().numpy()
#         xx, yy, zz = np.where(map > 0.5)
#
#         cube = mlab.points3d(xx, yy, zz,
#                              mode="cube",
#                              color=(0, 1, 0),
#                              scale_factor=1, transparent=True, opacity=0.5)
#
#         xx, yy, zz = np.where(label[0].detach().cpu().numpy() >= 0)
#
#         cube = mlab.points3d(xx, yy, zz,
#                              mode="cube",
#                              color=(0, 0, 0),
#                              scale_factor=1, transparent=True, opacity=0)
#         mlab.outline()
#
#         xx, yy, zz = np.where(label[0].detach().cpu().numpy() > 0.5)
#
#         cube = mlab.points3d(xx, yy, zz,
#                              mode="cube",
#                              color=(0, 0, 1),
#                              scale_factor=1, opacity=0.5)
#
#         # mlab.outline()
#         plot_vertices = ((output_vertices[0] * voxel_height + voxel_height - 1) / 2).data.detach().cpu().numpy()
#         cube = mlab.points3d(plot_vertices[:, 0].tolist(), plot_vertices[:, 1].tolist(), plot_vertices[:, 2].tolist(),
#                              mode="cube",
#                              color=(1, 0, 0),
#                              scale_factor=1, opacity=0.5)
#         # mlab.outline()
#         #
#         mlab.show()
#         plt.figure()
#         plt.plot(range(i + 1), losses, 'r-')
#         plt.savefig("loss.png")
#         plt.show()
from geoshape.models import ShapeDeformNet
model = ShapeDeformNet(voxel_width, voxel_height, voxel_depth, 10, 2)
# model = LVShapeNet(voxel_width, voxel_height, voxel_depth, 10)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
from tqdm import tqdm
pbar = tqdm(range(1000000))
image = label.unsqueeze(0)
for i in pbar:
    # print(vertices.shape, facets.shape)
    init_mask0, nodes0, init_mask1, nodes1, init_mask2, nodes2 = model(image)

    # mse_loss = loss(pred, label)
    # print(mse_loss.item())
    # mse_loss.backward()
    loss = (image - init_mask0).pow(2).mean()
    loss += (image - init_mask1).pow(2).mean()
    loss += (image - init_mask2).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print("loss", i, loss.item())
    # print((vertices[0] * 64 + 63) / 2)
    # print(i, vertices.grad[0])
    losses.append(loss.item())
    torch.save(model.state_dict(), "CP.pth")
    pbar.set_description("Loss: {}".format(loss.item()))
    # np_vertices = ((output_vertices[0] * voxel_depth + voxel_height - 1) / 2).detach().cpu().numpy()
    if i % 1000 == 0:
        plot = True
    else:
        plot = False
    if plot:
        map = init_mask[0, 0].detach().cpu().numpy()
        xx, yy, zz = np.where(map > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(1, 1, 0),
                             scale_factor=1, transparent=True, opacity=0.5)

        xx, yy, zz = np.where(label[0].detach().cpu().numpy() >= 0)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 0, 0),
                             scale_factor=1, transparent=True, opacity=0)
        mlab.outline()

        xx, yy, zz = np.where(lv_label > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 0, 1),
                             scale_factor=1, opacity=0.5)
        xx, yy, zz = np.where(lv_myo_label > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 1, 0),
                             scale_factor=1, opacity=0.5)
        xx, yy, zz = np.where(rv_label > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(1, 0, 0),
                             scale_factor=1, opacity=0.5)

        # mlab.outline()
        plot_vertices = ((output_vertices[0] * voxel_height + voxel_height - 1) / 2).data.detach().cpu().numpy()
        cube = mlab.points3d(plot_vertices[:, 0].tolist(), plot_vertices[:, 1].tolist(), plot_vertices[:, 2].tolist(),
                             mode="cube",
                             color=(1, 0, 0),
                             scale_factor=1, opacity=0.5)
        # mlab.outline()
        #
        mlab.show()
        plt.figure()
        plt.plot(range(i + 1), losses, 'r-')
        plt.savefig("loss_shapelv.png")
        plt.show()
