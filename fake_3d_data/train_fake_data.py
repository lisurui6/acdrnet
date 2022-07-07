import torch
import numpy as np
from rasterizor.voxelize import Voxelize
import nibabel as nib
from typing import Tuple
from scipy.ndimage import zoom
from torch.autograd import Variable
from matplotlib import pyplot as plt


voxel_width = 64
voxel_depth = 64
voxel_height = 64


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

#
# v0 = torch.from_numpy(np.array([0, 0, voxel_depth*5/6])).float().cuda()
# v1 = torch.from_numpy(np.array([voxel_width, voxel_height/2, voxel_depth*2/3])).float().cuda()
# v2 = torch.from_numpy(np.array([0, voxel_height/2, voxel_depth*2/3])).float().cuda()
# v3 = torch.from_numpy(np.array([voxel_width/2, voxel_height/2, voxel_depth])).float().cuda()

# # out voxel
# v0 = torch.from_numpy(np.array([0, 0, voxel_depth*5/6])).float().cuda()
# v1 = torch.from_numpy(np.array([voxel_width, voxel_height/2, voxel_depth*1/2])).float().cuda()
# v2 = torch.from_numpy(np.array([0, voxel_height, voxel_depth*2/3])).float().cuda()
# v3 = torch.from_numpy(np.array([voxel_width/2, voxel_height/2, voxel_depth])).float().cuda()

v0 = torch.from_numpy(np.array([64-0, 64-0, 64-voxel_depth*5/6])).float().cuda()
v1 = torch.from_numpy(np.array([64-voxel_width, 64-voxel_height/2, 64-voxel_depth*1/2])).float().cuda()
v2 = torch.from_numpy(np.array([64-0, 64-voxel_height, 64-voxel_depth*2/3])).float().cuda()
v3 = torch.from_numpy(np.array([64-voxel_width/2, 64-voxel_height/2, 64-voxel_depth])).float().cuda()


# in voxel
# v0 = torch.from_numpy(np.array([0, 0, voxel_height * 1 / 6])).float().cuda()
# v1 = torch.from_numpy(np.array([voxel_width, voxel_depth / 2, voxel_height * 1 / 4])).float().cuda()
# v2 = torch.from_numpy(np.array([0, voxel_depth, voxel_height * 1 / 3])).float().cuda()
# v3 = torch.from_numpy(np.array([voxel_width / 2, voxel_depth / 2, voxel_height])).float().cuda()


# small tetra
# x = 5
# v0 = torch.from_numpy(np.array([voxel_width / 2, voxel_depth / 2 - x, voxel_height * 2/3 - x])).float().cuda()
# v1 = torch.from_numpy(np.array([voxel_width / 2 + x, voxel_depth / 2 + x, voxel_height * 2/3 - x])).float().cuda()
# v2 = torch.from_numpy(np.array([voxel_width / 2 - x, voxel_depth / 2, voxel_height * 2/3 - x])).float().cuda()
# v3 = torch.from_numpy(np.array([voxel_width / 2, voxel_depth / 2, voxel_height * 2/3])).float().cuda()


image = nib.load("image.nii.gz").get_data()
if image.ndim == 4:
    image = np.squeeze(image, axis=-1).astype(np.int16)
image = image.astype(np.float32)
image = resize_image(image, (voxel_width, voxel_depth, voxel_height), 0)
# image = np.transpose(image, (2, 0, 1))
image = rescale_intensity(image, (1.0, 99.0))
image = np.expand_dims(image, 0)
image = torch.from_numpy(image).float()  # (1, w, h, d)

label = np.load("label_1.npy")
label = torch.from_numpy(label).unsqueeze(0).float().cuda()  # (1, w, h, d)

voxeliser = Voxelize(voxel_width, voxel_depth, voxel_height, eps=1e-4, eps_in=100)
vertices = torch.stack([v0, v1, v2, v3], dim=0).unsqueeze(0).float().cuda()  # (1, 4, 3)
print(vertices)
vertices[..., 0] = (vertices[..., 0] - voxel_width / 2 + 1/2) / voxel_width * 2
vertices[..., 1] = (vertices[..., 1] - voxel_depth / 2 + 1/2) / voxel_depth * 2
vertices[..., 2] = (vertices[..., 2] - voxel_height / 2 + 1/2) / voxel_height * 2
batch_size = 1
vertices = vertices.repeat(batch_size, 1, 1)
vertices = Variable(vertices, requires_grad=True)
facets = torch.from_numpy(np.array([0, 1, 2, 3])).unsqueeze(0).cuda()
facets = facets.repeat(batch_size, 1).unsqueeze(1)
# loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam([vertices], lr=1e-4)

losses = []
for i in range(1000000):
    # print(vertices.shape, facets.shape)
    pred = voxeliser(vertices, facets)

    # mse_loss = loss(pred, label)
    # print(mse_loss.item())
    # mse_loss.backward()
    loss = (label - pred).pow(2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss", i, loss.item())
    print((vertices[0] * 64 + 63) / 2)
    print(i, vertices.grad[0])
    losses.append(loss.item())
    np_vertices = ((vertices[0] * 64 + 63) / 2).detach().cpu().numpy()
    print(is_on_the_same_plane(np_vertices[0, :], np_vertices[1, :], np_vertices[2, :], np_vertices[3, :]))
    print()
    if i % 10000 == 0:
        plot = True
    else:
        plot = False
    if plot:
        from mayavi import mlab
        map = pred[0].detach().cpu().numpy()
        xx, yy, zz = np.where(map > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 1, 0),
                             scale_factor=1, transparent=True, opacity=0.1)

        xx, yy, zz = np.where(label[0].detach().cpu().numpy() >= 0)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 0, 0),
                             scale_factor=1, transparent=True, opacity=0)
        mlab.outline()

        xx, yy, zz = np.where(label[0].detach().cpu().numpy() > 0.5)

        cube = mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 0, 0),
                             scale_factor=1,)
        xx = [11, 13, 12, 14, 16, 11, 13, 15, 17, 19, 12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21, 23, 12, 14, 16, 18, 20, 22, 24, 11, 13, 15, 17, 19, 21, 23, 12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21, 23, 12, 14, 16, 18, 20, 11, 13, 15, 17, 12, 14, 11]
        yy = [15, 15, 18, 18, 18, 21, 21, 21, 21, 21, 24, 24, 24, 24, 24, 24, 27, 27, 27, 27, 27, 27, 27, 30, 30, 30, 30, 30, 30, 30, 33, 33, 33, 33, 33, 33, 33, 36, 36, 36, 36, 36, 36, 39, 39, 39, 39, 39, 39, 39, 42, 42, 42, 42, 42, 45, 45, 45, 45, 48, 48, 51]
        zz = [30, 33, 32, 35, 38, 31, 34, 37, 40, 43, 33, 36, 39, 42, 45, 48, 32, 35, 38, 41, 44, 47, 50, 34, 37, 40, 43, 46, 49, 52, 33, 36, 39, 42, 45, 48, 51, 35, 38, 41, 44, 47, 50, 34, 37, 40, 43, 46, 49, 52, 36, 39, 42, 45, 48, 35, 38, 41, 44, 37, 40, 36]
        # cube = mlab.points3d([4.8, 60.5, 5.29, 28.31], [3.97,32.03,64.83,29.48], [60.38,36.43,50.06,48],
        #                      mode="cube",
        #                      color=(1, 0, 0),
        #                      scale_factor=1,)
        # mlab.outline()
        plot_vertices = ((vertices[0] * 64 + 63) / 2).data.detach().cpu().numpy()
        cube = mlab.points3d(plot_vertices[:, 0].tolist(), plot_vertices[:, 1].tolist(), plot_vertices[:, 2].tolist(),
                             mode="cube",
                             color=(1, 0, 0),
                             scale_factor=1,)
        mlab.outline()

        mlab.show()
        plt.figure()
        plt.plot(range(i + 1), losses, 'r-')
        plt.show()


    # vertices.data -= 1e-9*vertices.grad.data
    #
    # # Manually zero the gradients after updating weights
    # vertices.grad.data.zero_()
    # vertices.grad.data.zero_()
