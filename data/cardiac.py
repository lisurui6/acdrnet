from torch.utils.data import Dataset
from imageio import imread
import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
from matplotlib import pyplot as plt
from data import transforms



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
        for i in range(2):
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


from scipy import ndimage


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