import torch
import numpy as np
import nibabel as nib
from typing import Tuple
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from mayavi import mlab
import torch.nn.functional as F
from geoshape.layers import Grad3D
import torch.nn as nn
from geoshape.models import ShapeDeformNet, LiteShapeDeformNet, MidShapeDeformNet

voxel_width = 64  # x
voxel_depth = 64  # y
voxel_height = 32  # z


def save_masks(init_masks, affine_masks, deform_masks, output_dir, image_path, iter, image, label):
    init_mask = torch.cat(init_masks, dim=1).squeeze(0).detach().cpu().numpy()
    affine_mask = torch.cat(affine_masks, dim=1).squeeze(0).detach().cpu().numpy()
    deform_mask = torch.cat(deform_masks, dim=1).squeeze(0).detach().cpu().numpy()

    nim = nib.load(str(image_path))
    # Transpose and crop the segmentation to recover the original size
    for predicted, prefix in zip([init_mask, affine_mask, deform_mask], ["init", "affine", "deform"]):

        # map back to original size
        final_predicted = np.zeros((init_mask[0].shape[0], init_mask[0].shape[1], init_mask[0].shape[2]))
        # print(predicted.shape, final_predicted.shape)
        final_predicted[predicted[2] > 0.5] = 3
        final_predicted[predicted[1] > 0.5] = 2
        final_predicted[predicted[0] > 0.5] = 1

        final_predicted = np.transpose(final_predicted, [1, 2, 0])

        nim2 = nib.Nifti1Image(final_predicted, nim.affine)
        nim2.header['pixdim'] = nim.header['pixdim']
        output_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, '{0}/{1}_{2}_seg.nii.gz'.format(str(output_dir), iter, prefix))

    image = image.detach().cpu().numpy().squeeze(0)
    final_image = np.transpose(image, [1, 2, 0])
    # print(final_image.shape)
    nim2 = nib.Nifti1Image(final_image, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/image.nii.gz'.format(str(output_dir)))
    # shutil.copy(str(input_path), str(output_dir.joinpath("image.nii.gz")))
    label = label[0]
    final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
    label = label.detach().cpu().numpy()
    for i in range(label.shape[0]):
        final_label[label[i, :, :, :] == 1.0] = i + 1

    final_label = np.transpose(final_label, [1, 2, 0])
    # print(final_label.shape)
    nim2 = nib.Nifti1Image(final_label, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nim2, '{0}/label.nii.gz'.format(str(output_dir)))


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


from pathlib import Path
# image_path = Path("lvsa_ED.nii.gz")
# label_path = Path("LVSA_seg_ED.nii.gz")
image_path = Path("lvsa_SR_ED.nii.gz")
label_path = Path("seg_lvsa_SR_ED.nii.gz")
# image = nib.load(image_path).get_data()
image = nib.load(str(image_path)).get_data()
if image.ndim == 4:
    image = np.squeeze(image, axis=-1).astype(np.int16)
image = image.astype(np.float32)
image = resize_image(image, (voxel_width, voxel_depth, voxel_height), 0)
# image = np.transpose(image, (2, 0, 1))
image = rescale_intensity(image, (1.0, 99.0))
image = np.expand_dims(image, 0)
image = torch.from_numpy(image).float().cuda()

# label = nib.load("seg_lvsa_SR_ED.nii.gz").get_data()
label = nib.load(label_path).get_data()
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
opacity = 1
xx, yy, zz = np.where(label[0].detach().cpu().numpy() >= 0.5)
cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 0, 1),
                     scale_factor=1, transparent=True, opacity=opacity)
xx, yy, zz = np.where(label[1].detach().cpu().numpy() >= 0.5)
cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 1, 0),
                     scale_factor=1, transparent=True, opacity=opacity)
xx, yy, zz = np.where(label[2].detach().cpu().numpy() >= 0.5)
cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(1, 0, 0),
                     scale_factor=1, transparent=True, opacity=opacity)
mlab.show()

model = ShapeDeformNet(voxel_width=voxel_width, voxel_depth=voxel_depth, voxel_height=voxel_height, num_lv_slices=32, num_extra_slices=3)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
from tqdm import tqdm
pbar = tqdm(range(1000000))
label = label.unsqueeze(0)
torch.autograd.set_detect_anomaly(True)
flow_grad_loss = Grad3D(penalty="l2")
for i in pbar:
    [init_mask0, init_mask1, init_mask2], \
    [affine_mask0, affine_mask1, affine_mask2], \
    [deform_mask0, deform_mask1, deform_mask2], \
    [nodes0, nodes1, nodes2], flow, preint_flow = model(image)
    loss = 0
    loss += (label[:, 0] - init_mask0).pow(2).mean()
    loss += (label[:, 1] - init_mask1).pow(2).mean()
    loss += (label[:, 2] - init_mask2).pow(2).mean()
    loss += (label[:, 0] - affine_mask0).pow(2).mean()
    loss += (label[:, 1] - affine_mask1).pow(2).mean()
    loss += (label[:, 2] - affine_mask2).pow(2).mean()
    loss += (label[:, 0] - deform_mask0).pow(2).mean()
    loss += (label[:, 1] - deform_mask1).pow(2).mean()
    loss += (label[:, 2] - deform_mask2).pow(2).mean()
    loss += 100 * flow_grad_loss(flow)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    torch.save(model.state_dict(), "CP.pth")
    pbar.set_description("Loss: {}".format(loss.item()))
    if i % 1000 == 0:
        plot = True
    else:
        plot = False
    if plot:
        opacity = 0.999
        save_masks(
            [init_mask0, init_mask1, init_mask2], [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2],
            output_dir=Path(__file__).parent, image_path=image_path, iter=i, image=image, label=label
        )
        for masks, prefix in zip([[init_mask0, init_mask1, init_mask2], [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2]], ["init", "affine", "deform"]):
            print(prefix)
            map = masks[1][0, 0].detach().cpu().numpy()
            xx, yy, zz = np.where(map > 0.5)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(1, 1, 0),
                                 scale_factor=1, transparent=True, opacity=opacity)

            map = masks[0][0, 0].detach().cpu().numpy()
            xx, yy, zz = np.where(map > 0.5)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(0, 1, 1),
                                 scale_factor=1, transparent=True, opacity=opacity)

            map = masks[2][0, 0].detach().cpu().numpy()
            xx, yy, zz = np.where(map > 0.5)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(1, 0, 1),
                                 scale_factor=1, transparent=True, opacity=opacity)

            # xx, yy, zz = np.where(label[0, 0].detach().cpu().numpy() >= 0)
            #
            # cube = mlab.points3d(xx, yy, zz,
            #                      mode="cube",
            #                      color=(0, 0, 0),
            #                      scale_factor=1, transparent=True, opacity=0)
            # mlab.outline()

            # mlab.show()
            xx, yy, zz = np.where(label[0, 0].detach().cpu().numpy() >= 0)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(0, 0, 0),
                                 scale_factor=1, transparent=True, opacity=0)
            mlab.outline()

            xx, yy, zz = np.where(lv_label > 0.5)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(0, 0, 1),
                                 scale_factor=1, opacity=1)
            xx, yy, zz = np.where(lv_myo_label > 0.5)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(0, 1, 0),
                                 scale_factor=1, opacity=1)
            xx, yy, zz = np.where(rv_label > 0.5)

            cube = mlab.points3d(xx, yy, zz,
                                 mode="cube",
                                 color=(1, 0, 0),
                                 scale_factor=1, opacity=1)

            # mlab.outline()
            # plot_vertices = ((nodes0[0] * voxel_height + voxel_height - 1) / 2).data.detach().cpu().numpy()
            # cube = mlab.points3d(plot_vertices[:, 0].tolist(), plot_vertices[:, 1].tolist(), plot_vertices[:, 2].tolist(),
            #                      mode="cube",
            #                      color=(1, 1, 0),
            #                      scale_factor=1, opacity=0.5)
            #
            # plot_vertices = ((nodes2[0] * voxel_height + voxel_height - 1) / 2).data.detach().cpu().numpy()
            # cube = mlab.points3d(plot_vertices[:, 0].tolist(), plot_vertices[:, 1].tolist(), plot_vertices[:, 2].tolist(),
            #                      mode="cube",
            #                      color=(1, 1, 0),
            #                      scale_factor=1, opacity=0.5)
            # mlab.outline()
            #
            mlab.show()
        plt.figure()
        plt.plot(range(i + 1), losses, 'r-')
        plt.savefig("loss_shape.png")
        plt.show()
