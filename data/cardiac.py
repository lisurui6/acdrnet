from torch.utils.data import Dataset
from imageio import imread
import os
import cv2
import numpy as np
import torch
import matplotlib
# matplotlib.use('TkAgg')
from pathlib import Path
from matplotlib import pyplot as plt
from data import transforms
import nibabel as nib
from scipy import ndimage
from typing import Tuple
from scipy.ndimage import zoom
import SimpleITK as sitk


class Cardiac3dDataset(Dataset):
    def __init__(self, data_dir: Path, voxel_width, voxel_height, voxel_depth, save_dir: Path):
        self.data_dir = data_dir
        image_list = []
        label_list = []
        image_prefix = "lvsa_SR"
        label_prefix = "seg_lvsa_SR"
        for subject in os.listdir(str(data_dir)):
            subject_dir = data_dir.joinpath(subject)
            if subject_dir.joinpath(f"{image_prefix}_ED.nii.gz").exists() and subject_dir.joinpath(f"{label_prefix}_ED.nii.gz").exists():
                image_list.append(subject_dir.joinpath(f"{image_prefix}_ED.nii.gz"))
                label_list.append(subject_dir.joinpath(f"{label_prefix}_ED.nii.gz"))
            if subject_dir.joinpath(f"{image_prefix}_ES.nii.gz").exists() and subject_dir.joinpath(f"{label_prefix}_ES.nii.gz").exists():
                image_list.append(subject_dir.joinpath(f"{image_prefix}_ES.nii.gz"))
                label_list.append(subject_dir.joinpath(f"{label_prefix}_ES.nii.gz"))

        self.image_list = image_list
        self.label_list = label_list
        self.voxel_width = voxel_width
        self.voxel_height = voxel_height
        self.voxel_depth = voxel_depth
        self.save_dir = save_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_path = self.image_list[item]
        label_path = self.label_list[item]
        image = nib.load(str(image_path)).get_data()
        if image.ndim == 4:
            image = np.squeeze(image, axis=-1).astype(np.int16)
        image = image.astype(np.float32)
        image = self.resize_image(image, (self.voxel_width, self.voxel_height, self.voxel_depth), 0)
        # image = np.transpose(image, (2, 0, 1))
        image = self.rescale_intensity(image, (1.0, 99.0))

        label = self.read_label(label_path)
        if self.save_dir is not None:
            self.save(image, label, item)

        image = np.expand_dims(image, 0)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label

    def save(self, image: np.ndarray, label: np.ndarray, index: int):
        if index % 100 == 0:
            nim = nib.load(str(self.image_list[index]))
            image = np.transpose(image, [1, 2, 0])
            nim2 = nib.Nifti1Image(image, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_path = self.save_dir.joinpath("save", "image_{}.nii.gz".format(index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))

            final_label = np.zeros((label.shape[1], label.shape[2], label.shape[3]))
            for i in range(label.shape[0]):
                final_label[label[i, :, :, :] == 1.0] = i + 1
            final_label = np.transpose(final_label, [1, 2, 0])
            nim2 = nib.Nifti1Image(final_label, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            output_path = self.save_dir.joinpath("save", "label_{}.nii.gz".format(index))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))


    @staticmethod
    def resize_image(image: np.ndarray, target_shape: Tuple, order: int):
        image_shape = image.shape
        factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
        output = zoom(image, factors, order=order)
        return output

    @staticmethod
    def rescale_intensity(image, thres=(1.0, 99.0)):
        """ Rescale the image intensity to the range of [0, 1] """
        val_l, val_h = np.percentile(image, thres)
        image2 = image
        image2[image < val_l] = val_l
        image2[image > val_h] = val_h
        image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
        return image2

    def read_label(self, label_path: Path) -> np.ndarray:
        label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
        label = np.transpose(label, axes=(2, 1, 0))
        if label.ndim == 4:
            label = np.squeeze(label, axis=-1).astype(np.int16)
        label = label.astype(np.float32)
        label[label == 4] = 3
        # if crop:
        #     label = Torch2DSegmentationDataset.crop_3D_image(label, cx, cy, feature_size, cz, n_slices)
        # else:
        label = self.resize_image(label, (self.voxel_width, self.voxel_height, self.voxel_depth), 0)
        X, Y, Z = label.shape

        labels = []
        for i in range(1, 4):
            # blank_image = np.zeros((feature_size, feature_size, n_slices))
            blank_image = np.zeros((X, Y, Z))

            blank_image[label == i] = 1
            labels.append(blank_image)
        label = np.array(labels)
        # label = np.transpose(label, (0, 3, 1, 2))
        return label


class CardiacDataset(Dataset):
    def __init__(self, data_path: Path, ann='train', transformations=None, train_size=1500, test_size=666):
        self.img_path = data_path.joinpath("image")
        self.label_path = data_path.joinpath("label")
        self.transformations = transformations
        self.ann = ann

        self.img_list = os.listdir(str(self.img_path))
        self.img_list.sort()
        self.mask_list = os.listdir(str(self.label_path))
        self.mask_list.sort()

        if self.ann == 'train':
            self.img_list = self.img_list[0: train_size]
            self.mask_list = self.mask_list[0: train_size]
        else:
            self.img_list = self.img_list[train_size:train_size + test_size]
            self.mask_list = self.mask_list[train_size:train_size + test_size]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_file = self.img_list[item]
        mask_file = self.mask_list[item]

        image = cv2.imread(str(self.img_path) + '/' + img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)

        mask = cv2.imread(str(self.label_path) + '/' + mask_file)

        if self.transformations is not None:
            image, mask = self.transformations(image, mask)
        image = torch.from_numpy(np.array(image)).float()
        image = image.unsqueeze(0)
        mask = torch.from_numpy(np.array(mask)).permute(2, 0, 1).float()
        mask[mask > 0] = 1  # (3, W, H)
        mask[1, :, :][mask[0, :, :] > 0] = 1
        mask[1, :, :][mask[1, :, :] > 0] = 1

        dmap = signed_distance(mask)  # (3, W, H)
        g_maps = []
        for i in range(3):
            g_x, g_y = np.gradient(dmap[i, :, :])
            g_x = np.multiply(dmap[i, :, :] / 128, -g_x)
            g_y = np.multiply(dmap[i, :, :] / 128, -g_y)
            g_x[abs(dmap[i, :, :]) > 50] = 0
            g_y[abs(dmap[i, :, :]) > 50] = 0

            g_map = np.array([g_y, g_x], dtype=np.float32)
            g_maps.append(g_map)
        g_map = np.concatenate(g_maps, axis=0)
        g_map = torch.from_numpy(g_map)  # (3, 2, W, H)

        return image, mask, g_map




def signed_distance(mask):
    """Mask shape = (3, H, W)"""
    distances = []
    f = np.uint8(mask[0, :, :] > 0.5)
    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))
    distances.append(distance)

    # f = np.zeros_like(mask[0, :, :])
    # f[mask[0, :, :] > 0.5] = 1
    # f[mask[1, :, :] > 0.5] = 1
    # f = np.uint8(f)

    f = np.uint8(mask[1, :, :] > 0.5)

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))
    distances.append(distance)

    f = np.uint8(mask[2, :, :] > 0.5)
    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))
    distances.append(distance)
    distances = np.stack(distances)

    return distances


# val_trans = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((128, 128)),
#     # transforms.ToTensor(),
#     # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
#     # transforms.NormalizeInstance()
#     # transforms.Normalize(MEAN, STD),
# ])
#
# train_ds = CardiacDataset(Path(__file__).parent.parent.joinpath("rbh_2d_data"), "train", transformations=val_trans)
# image, mask, d_map = train_ds[0]
#
# from matplotlib import pyplot as plt
#
# image, mask = image.numpy(), mask.numpy()
#
# plt.imshow(image)
# plt.show()
#
# plt.imshow(d_map[0, :, :])
# plt.show()
#
# plt.imshow(d_map[1, :, :])
# plt.show()

# plt.imshow(d_map[2, :, :])
# plt.show()