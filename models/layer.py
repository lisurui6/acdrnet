import torch
import torch.nn.functional as F


class DistanceTransformLayer(torch.nn.Module):
    def __init__(self, feature_size: int):
        """Kernel size has to an odd number"""
        super().__init__()
        # with torch.no_grad():
        xx, yy = torch.meshgrid(torch.arange(feature_size), torch.arange(feature_size))
        self.xx = xx.float().cuda().detach()
        self.yy = yy.float().cuda().detach()

    def forward(self, feature_map):
        """Feature map is (B, 1, H, W), output is in (1, H, W) too. """

        output = torch.empty(feature_map.shape).cuda()
        # with torch.no_grad():
        x = self.xx.repeat(feature_map.shape[0], feature_map.shape[1], 1, 1)
        y = self.yy.repeat(feature_map.shape[0], feature_map.shape[1], 1, 1)
        xx = torch.empty(feature_map.shape, requires_grad=False).cuda()
        yy = torch.empty(feature_map.shape, requires_grad=False).cuda()
        # feature_map_copy = torch.empty_like(feature_map).cuda()
        # feature_map_copy.copy_(feature_map)
        bdry = torch.where(feature_map > 0.5)
        for i in range(0, feature_map.shape[2]):
            xx.copy_(x).sub_(i).square_()
            # xx = (x - i).detach().square().float()
            # xx = xx_init.add_(x).sub_(i).square_()
            for j in range(0, feature_map.shape[3]):
                # print(i, j)
                # with torch.no_grad():
                yy.copy_(y).sub_(j).square_()
                yy.add_(xx).sqrt_()

                # yy = (y - j).detach().square().float()
                # yy.sub_(feature_map)
                output[:, :, i, j] = torch.min(yy[bdry])
                # output[:, :, i, j] = torch.amin(yy + feature_map, dim=[2, 3])
                # output[:, :, i, j] = torch.amax(-torch.sqrt(xx+yy)-feature_map, dim=[2, 3])

        return output




class DistanceTransformLayer2(torch.nn.Module):
    def __init__(self, feature_size: int):
        """Kernel size has to an odd number"""
        super().__init__()
        with torch.no_grad():
            xx, yy = torch.meshgrid(torch.arange(feature_size), torch.arange(feature_size))
            xx = xx.float().cuda().view(-1)
            yy = yy.float().cuda().view(-1)
            grid = torch.stack([xx, yy], dim=1)
            self.cdist = torch.chunk(torch.cdist(grid, grid, p=2), feature_size*feature_size)
            print(grid.shape)

    def forward(self, feature_map):
        """Feature map is (B, 1, H, W), output is in (1, H, W) too. """

        output = torch.empty(feature_map.shape).cuda()

        for i in range(0, feature_map.shape[2]):
            with torch.no_grad():
                cdist = self.cdist[i].cuda().view(feature_map.shape[2], feature_map.shape[3])
                cdist = cdist.repeat(feature_map.shape[0], feature_map.shape[1], 1, 1)
            for j in range(0, feature_map.shape[3]):
                print(i, j)
                # yy = (y - j).detach().square().float()
                # cdist_index = i * feature_map.shape[2] + j
                output[:, :, i, j] = torch.amax(cdist.mul_(-1).sub_(feature_map), dim=[2, 3])
                # output[:, :, i, j] = torch.amax(-torch.sqrt(xx+yy)-feature_map, dim=[2, 3])

        return output


class MinDistanceConvLayer2(torch.nn.Module):
    def __init__(self, kernel_size: int, n_channels: int, feature_size: int):
        """Kernel size has to an odd number"""
        super().__init__()
        self.kernel_size = kernel_size
        # kernel = torch.empty((self.kernel_size, self.kernel_size)).cuda()
        # center = (self.kernel_size - 1)//2 + 1
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         kernel[i, j] = torch.mul(i - center, i - center).float() + \
        #                             torch.mul(j - center, j - center).float()
        #
        # kernel = kernel.cuda().expand(feature_size, feature_size, n_channels, self.kernel_size, self.kernel_size)
        # self.kernel = kernel.permute(2, 0, 1, 3, 4)
        xx, yy = torch.meshgrid(torch.arange(feature_size), torch.arange(feature_size))
        self.xx = xx.cuda()
        self.yy = yy.cuda()

    def forward(self, feature_map):
        """Feature map is (B, 1, H, W), output is in (1, H, W) too. """
        border_size = (self.kernel_size - 1)//2
        # kernel = self.kernel.expand(feature_map.shape[0], -1, -1, -1, -1, -1)
        # padded = F.pad(feature_map, [border_size, border_size, border_size, border_size], "constant", 0)
        #
        # output = padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).mul_(-1).sub_(kernel).amax(dim=[4, 5])
        # output = output.contiguous().view(feature_map.shape)

        output = torch.empty(feature_map.shape).cuda()
        for b in range(feature_map.shape[0]):
            for c in range(feature_map.shape[1]):
                for i in range(0, feature_map.shape[2]):
                    for j in range(0, feature_map.shape[3]):
                # for i in range(border_size, border_size + feature_map.shape[1]):
                #     for j in range(border_size, border_size + feature_map.shape[2]):
                        # min_dist = torch.min(torch.where(
                        #     padded[b, :, i - border_size: i + border_size + 1, j - border_size: j + border_size + 1] > 0.5,
                        #     self.kernel, self.kernel + 1000
                        # ))
                        # sliced = padded[:, :, i - border_size: i + border_size + 1, j - border_size: j + border_size + 1]
                        # temp = torch.sub(-sliced, kernel)
                        # amax = torch.amax(temp, dim=[2, 3])
                        # output[:, :, i - border_size, j - border_size] = amax

                        temp = torch.max(
                            -torch.sqrt(((self.xx - i).square() + (self.yy - j).square()))-feature_map[b, c, :, :]
                        )
                        output[b, c, i, j] = temp

        return output


class DistanceTransformPooling(torch.nn.Module):
    """https://arxiv.org/pdf/1409.5403.pdf"""
    def __init__(self, n_channels, ):
        super().__init__()

    def forward(self):
        pass


class ExpActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-torch.pow(x, 2)) * 5000
        # p = torch.ones(x.shape).cuda() * 5000
        # return torch.pow(p, -torch.pow(x, 2))


class SpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        # should it be minus?
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(torch.nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


# # Test DTlayer
# import numpy as np
# from matplotlib import pyplot as plt
# feature_size = 128
# image = np.ones((feature_size, feature_size)) * 1000 ** 2
# xx, yy = np.meshgrid(np.arange(feature_size), np.arange(feature_size))
# dist_from_center = (xx - feature_size//2)**2 + (yy - feature_size//2)**2
# radius = int(feature_size*0.3)**2
# radius2 = (int(feature_size*0.3) + 2)**2
#
# circular_mask = (dist_from_center < radius2) & (dist_from_center > radius)
# image[circular_mask] = 0
# plt.imshow(image)
# plt.show()
# image = np.expand_dims(image, 0)
# image = np.expand_dims(image, 0)
#
# image_tensor = torch.from_numpy(image).cuda()
# layer = DistanceTransformLayer(feature_size)
# output = layer(image_tensor)
# output = output.squeeze().cpu().detach().numpy()
#
# plt.imshow(output)
# plt.show()
