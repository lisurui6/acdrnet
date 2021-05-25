from torch.utils.data import Dataset
from imageio import imread
import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


class BuildingsDataset(Dataset):
    def __init__(self,
                 data_path,
                 ann='train',
                 transformations=None,
                 train_size=100,
                 test_size=68):
        self.img_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.transformations = transformations
        self.ann = ann

        self.img_list = os.listdir(self.img_path)
        self.img_list.sort()
        self.mask_list = os.listdir(self.mask_path)
        self.mask_list.sort()

        if self.ann == 'train':
            self.img_list = self.img_list[0:train_size]
            self.mask_list = self.mask_list[0:train_size]
        else:
            self.img_list = self.img_list[train_size:train_size + test_size]
            self.mask_list = self.mask_list[train_size:train_size + test_size]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_file = self.img_list[item]
        mask_file = self.mask_list[item]

        image = cv2.imread(self.img_path + '/' + img_file)
        mask = cv2.imread(self.mask_path + '/' + mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask > 0] = 1
        # mask = np.expand_dims(mask, axis=0)

        if self.transformations is not None:
            image, mask = self.transformations(image, mask)
        dmap = signed_distance(mask)

        g_x, g_y = np.gradient(dmap)
        g_x = np.multiply(dmap / 128, -g_x)
        g_y = np.multiply(dmap / 128, -g_y)
        g_x[abs(dmap) > 20] = 0
        g_y[abs(dmap) > 20] = 0

        g_map = np.array([g_y, g_x], dtype=np.float32)
        g_map = torch.from_numpy(g_map)

        return image, mask, g_map


from scipy import ndimage


def signed_distance(mask):
    f = np.uint8(mask > 0.5)
    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))
    return distance
