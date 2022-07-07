import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import monai

# from models.networks.shape_3d import ShapeDeformCardiac3DCircleNet
from models.networks.shape_3d_smooth import ShapeDeformCardiac3DCircleNet
# from models.networks.shape_3d_smooth_lv import ShapeDeformCardiac3DCircleNet


from data.cardiac import Cardiac3dDataset
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrices import get_ap_scores, get_iou, get_f1_scores
from data import transforms
from models.loss_functions import curvature_loss, dist_loss
from torchvision.utils import make_grid
from models.loss_functions import Grad
import nibabel as nib

# --resume
# C:\Users\lisur\PycharmProjects\ACDRNet-master\run\buildings\rider_DEBUG_unet\experiment_16\checkpoint.pth.tar


def visualise_images(image, mask, pred_masks, affine_masks, global_step, output_dir: Path, prefix: str = "train", init_masks=None):
    # save image
    for i in range(0, 2):
        np_image = image[i].detach().cpu().numpy().squeeze()
        nim2 = nib.Nifti1Image(np_image, None)
        output_path = output_dir.joinpath(prefix, "step_" + str(global_step), "{}_image.nii.gz".format(i))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, str(output_path))

        displace_mask = torch.zeros((np_image.shape[0], np_image.shape[1], np_image.shape[2]))
        mask[i, 1, :, :][mask[i, 0, :, :] > 0.5] = 0

        for c in range(3):
            displace_mask[mask[i, c, ...] > 0.5] = c + 1
        np_mask = displace_mask.detach().cpu().numpy()
        nim2 = nib.Nifti1Image(np_mask, None)
        output_path = output_dir.joinpath(prefix, "step_" + str(global_step), "{}_label.nii.gz".format(i))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nim2, str(output_path))

        if init_masks is not None:
            displace_mask = torch.zeros((np_image.shape[0], np_image.shape[1], np_image.shape[2]))
            if len(init_masks) == 3:
                init_masks[1][i, ...][init_masks[0][i, ...] > 0.5] = 0
                for c in range(3):
                    displace_mask[init_masks[c][i, ...].squeeze(0) > 0.5] = c + 1
            else:
                displace_mask[init_masks[0][i, ...].squeeze(0) > 0.5] = 1

            np_mask = displace_mask.detach().cpu().numpy()

            nim2 = nib.Nifti1Image(np_mask, None)
            output_path = output_dir.joinpath(prefix, "step_" + str(global_step), "{}_init_mask.nii.gz".format(i))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))
        if affine_masks is not None:
            displace_mask = torch.zeros((np_image.shape[0], np_image.shape[1], np_image.shape[2]))
            affine_masks[1][i, ...][affine_masks[0][i, ...] > 0.5] = 0
            for c in range(3):
                displace_mask[affine_masks[c][i, ...].squeeze(0) > 0.5] = c + 1
            np_mask = displace_mask.detach().cpu().numpy()

            nim2 = nib.Nifti1Image(np_mask, None)
            output_path = output_dir.joinpath(prefix, "step_" + str(global_step), "{}_affine_mask.nii.gz".format(i))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))
        if pred_masks is not None:
            displace_mask = torch.zeros((np_image.shape[0], np_image.shape[1], np_image.shape[2]))
            pred_masks[1][i, ...][pred_masks[0][i, ...] > 0.5] = 0

            for c in range(3):
                displace_mask[pred_masks[c][i, ...].squeeze(0) > 0.5] = c + 1
            np_mask = displace_mask.detach().cpu().numpy()

            nim2 = nib.Nifti1Image(np_mask, None)
            output_path = output_dir.joinpath(prefix, "step_" + str(global_step), "{}_pred_mask.nii.gz".format(i))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nim2, str(output_path))


def plot_2(flow, image, mask, affine_masks, deformed_masks):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import torch.nn.functional as F

    def plot_grid(x, y, ax=None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x, y), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))
        ax.autoscale()

    # fig, ax = plt.subplots()
    n = 128
    # grid_x, grid_y = np.meshgrid(np.linspace(0, 127, n), np.linspace(0, 127, n))
    # plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
    # plt.show()
    # torch_grid = torch.stack([torch.from_numpy(grid_x), torch.from_numpy(grid_y)], dim=0).cuda().float().unsqueeze(0)
    # sample_grid = torch.stack([torch.from_numpy(grid_x), torch.from_numpy(grid_y)], dim=0).cuda().float().unsqueeze(0)
    vectors = [torch.arange(0, s, 2) for s in (128, 128)]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor).cuda()
    grid_x = grid[:, 0, :, :].squeeze().detach().cpu().numpy()
    grid_y = grid[:, 1, :, :].squeeze().detach().cpu().numpy()
    flow = F.interpolate(flow, size=(64, 64), mode='bilinear')
    for b in range(10):
        fig, ax = plt.subplots()

        plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
        # sample_grid[..., 1] = sample_grid[..., 1] * -1
        new_locs = grid + flow[b].unsqueeze(0)
        for i in range(2):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (128 - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        # Pxx = F.grid_sample(flow[b].unsqueeze(0), torch_grid).transpose(3, 2)
        # Pxx = F.grid_sample(flow[b].unsqueeze(0)[:, 0:1], sample_grid)
        # Pyy = F.grid_sample(flow[b].unsqueeze(0)[:, 1:2], sample_grid)
        # dP0 = torch.stack((Pxx, Pyy), -1).squeeze(0)
        # Pyy = F.grid_sample(flow[b, 1:2].unsqueeze(0), torch_grid).transpose(3, 2)
        # dist = torch_grid + dP0.permute([0, 2, 3, 1])
        # dist = F.grid_sample(sample_grid, new_locs)
        dist = new_locs

        # sample_grid[..., 1] = sample_grid[..., 1] * -1

        # dist[..., 1] = dist[..., 1] * -1
        dist = (dist / 2 + 0.5) * 127

        distx = dist.squeeze(0)[:, :, 0]
        disty = dist.squeeze(0)[:, :, 1]
        distx = distx.detach().cpu().numpy()
        disty = disty.detach().cpu().numpy()


        # plt.figure()
        # plt.imshow(image[0].squeeze(0).detach().cpu().numpy())
        # plt.figure()
        # displace_mask = torch.zeros_like(mask[b, :, :, :])
        # mask[b, 1, :, :][mask[b, 0, :, :] > 0.5] = 0
        # displace_mask[0, :, :][mask[b, 0, :, :] > 0.5] = 1
        # displace_mask[1, :, :][mask[b, 1, :, :] > 0.5] = 1
        # displace_mask[2, :, :][mask[b, 2, :, :] > 0.5] = 1
        # displace_mask[0, :, :][(mask[b, 0, :, :] < 0.5) & (mask[b, 1, :, :] < 0.5) & (mask[b, 2, :, :] < 0.5)] = 1
        # displace_mask[1, :, :][(mask[b, 0, :, :] < 0.5) & (mask[b, 1, :, :] < 0.5) & (mask[b, 2, :, :] < 0.5)] = 1
        # displace_mask[2, :, :][(mask[b, 0, :, :] < 0.5) & (mask[b, 1, :, :] < 0.5) & (mask[b, 2, :, :] < 0.5)] = 1

        displace_mask = torch.zeros_like(mask[b, :, :, :])
        affine_masks[1][b, 0, :, :][affine_masks[0][b, 0, :, :] >= 0.5] = 0
        displace_mask[0, :, :][affine_masks[0][b, 0, :, :] >= 0.5] = 1
        displace_mask[1, :, :][affine_masks[1][b, 0, :, :] >= 0.5] = 1
        displace_mask[2, :, :][affine_masks[2][b, 0, :, :] >= 0.5] = 1
        displace_mask[0, :, :][(affine_masks[0][b, 0, :, :] < 0.5) & (affine_masks[1][b, 0, :, :] < 0.5) & (affine_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[1, :, :][(affine_masks[0][b, 0, :, :] < 0.5) & (affine_masks[1][b, 0, :, :] < 0.5) & (affine_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[2, :, :][(affine_masks[0][b, 0, :, :] < 0.5) & (affine_masks[1][b, 0, :, :] < 0.5) & (affine_masks[2][b, 0, :, :] < 0.5)] = 1

        displace_mask = displace_mask.permute(1, 2, 0)
        plt.imshow(displace_mask.detach().cpu().numpy())
        plot_grid(distx, disty, ax=ax, color="C0")

        # plt.show()
        fig, ax = plt.subplots()

        plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
        displace_mask = torch.zeros_like(mask[b, :, :, :])
        deformed_masks[1][b, 0, :, :][deformed_masks[0][b, 0, :, :] >= 0.5] = 0
        displace_mask[0, :, :][deformed_masks[0][b, 0, :, :] >= 0.5] = 1
        displace_mask[1, :, :][deformed_masks[1][b, 0, :, :] >= 0.5] = 1
        displace_mask[2, :, :][deformed_masks[2][b, 0, :, :] >= 0.5] = 1
        displace_mask[0, :, :][(deformed_masks[0][b, 0, :, :] < 0.5) & (deformed_masks[1][b, 0, :, :] < 0.5) & (deformed_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[1, :, :][(deformed_masks[0][b, 0, :, :] < 0.5) & (deformed_masks[1][b, 0, :, :] < 0.5) & (deformed_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[2, :, :][(deformed_masks[0][b, 0, :, :] < 0.5) & (deformed_masks[1][b, 0, :, :] < 0.5) & (deformed_masks[2][b, 0, :, :] < 0.5)] = 1

        displace_mask = displace_mask.permute(1, 2, 0)
        plt.imshow(displace_mask.detach().cpu().numpy())
        plot_grid(distx, disty, ax=ax, color="C0")

        fig, ax = plt.subplots()
        plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
        plt.imshow(image[b].squeeze(0).detach().cpu().numpy())
        plot_grid(distx, disty, ax=ax, color="C0")
        plt.show()




def plot(flow, image, mask, affine_masks, deformed_masks):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import torch.nn.functional as F

    def plot_grid(x, y, ax=None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x, y), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, **kwargs))
        ax.add_collection(LineCollection(segs2, **kwargs))
        ax.autoscale()

    # fig, ax = plt.subplots()
    n = 80
    grid_x, grid_y = np.meshgrid(np.linspace(0, 127, n), np.linspace(0, 127, n))
    # plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
    # plt.show()
    torch_grid = torch.stack([torch.from_numpy(grid_x), torch.from_numpy(grid_y)], dim=2).cuda().float().unsqueeze(0)
    sample_grid = torch.stack([torch.from_numpy(grid_x), torch.from_numpy(grid_y)], dim=2).cuda().float().unsqueeze(0)

    for i in range(2):
        torch_grid[..., i] = 2 * (torch_grid[..., i] / (128 - 1) - 0.5)

    for i in range(2):
        sample_grid[..., i] = 2 * (sample_grid[..., i] / (128 - 1) - 0.5)
    for b in range(10):
        fig, ax = plt.subplots()

        plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
        # sample_grid[..., 1] = sample_grid[..., 1] * -1

        # Pxx = F.grid_sample(flow[b].unsqueeze(0), torch_grid).transpose(3, 2)
        Pxx = F.grid_sample(flow[b].unsqueeze(0)[:, 0:1], sample_grid)
        Pyy = F.grid_sample(flow[b].unsqueeze(0)[:, 1:2], sample_grid)
        dP0 = torch.stack((Pxx, Pyy), -1).squeeze(0)
        # Pyy = F.grid_sample(flow[b, 1:2].unsqueeze(0), torch_grid).transpose(3, 2)
        # dist = torch_grid + dP0.permute([0, 2, 3, 1])
        dist = sample_grid + dP0
        # sample_grid[..., 1] = sample_grid[..., 1] * -1

        # dist[..., 1] = dist[..., 1] * -1
        dist = (dist / 2 + 0.5) * 127

        distx = dist.squeeze(0)[:, :, 0]
        disty = dist.squeeze(0)[:, :, 1]
        distx = distx.detach().cpu().numpy()
        disty = disty.detach().cpu().numpy()


        # plt.figure()
        # plt.imshow(image[0].squeeze(0).detach().cpu().numpy())
        # plt.figure()
        # displace_mask = torch.zeros_like(mask[b, :, :, :])
        # mask[b, 1, :, :][mask[b, 0, :, :] > 0.5] = 0
        # displace_mask[0, :, :][mask[b, 0, :, :] > 0.5] = 1
        # displace_mask[1, :, :][mask[b, 1, :, :] > 0.5] = 1
        # displace_mask[2, :, :][mask[b, 2, :, :] > 0.5] = 1
        # displace_mask[0, :, :][(mask[b, 0, :, :] < 0.5) & (mask[b, 1, :, :] < 0.5) & (mask[b, 2, :, :] < 0.5)] = 1
        # displace_mask[1, :, :][(mask[b, 0, :, :] < 0.5) & (mask[b, 1, :, :] < 0.5) & (mask[b, 2, :, :] < 0.5)] = 1
        # displace_mask[2, :, :][(mask[b, 0, :, :] < 0.5) & (mask[b, 1, :, :] < 0.5) & (mask[b, 2, :, :] < 0.5)] = 1

        displace_mask = torch.zeros_like(mask[b, :, :, :])
        affine_masks[1][b, 0, :, :][affine_masks[0][b, 0, :, :] >= 0.5] = 0
        displace_mask[0, :, :][affine_masks[0][b, 0, :, :] >= 0.5] = 1
        displace_mask[1, :, :][affine_masks[1][b, 0, :, :] >= 0.5] = 1
        displace_mask[2, :, :][affine_masks[2][b, 0, :, :] >= 0.5] = 1
        displace_mask[0, :, :][(affine_masks[0][b, 0, :, :] < 0.5) & (affine_masks[1][b, 0, :, :] < 0.5) & (affine_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[1, :, :][(affine_masks[0][b, 0, :, :] < 0.5) & (affine_masks[1][b, 0, :, :] < 0.5) & (affine_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[2, :, :][(affine_masks[0][b, 0, :, :] < 0.5) & (affine_masks[1][b, 0, :, :] < 0.5) & (affine_masks[2][b, 0, :, :] < 0.5)] = 1

        displace_mask = displace_mask.permute(1, 2, 0)
        plt.imshow(displace_mask.detach().cpu().numpy())
        plot_grid(distx, disty, ax=ax, color="C0")

        # plt.show()
        fig, ax = plt.subplots()

        plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
        displace_mask = torch.zeros_like(mask[b, :, :, :])
        deformed_masks[1][b, 0, :, :][deformed_masks[0][b, 0, :, :] >= 0.5] = 0
        displace_mask[0, :, :][deformed_masks[0][b, 0, :, :] >= 0.5] = 1
        displace_mask[1, :, :][deformed_masks[1][b, 0, :, :] >= 0.5] = 1
        displace_mask[2, :, :][deformed_masks[2][b, 0, :, :] >= 0.5] = 1
        displace_mask[0, :, :][(deformed_masks[0][b, 0, :, :] < 0.5) & (deformed_masks[1][b, 0, :, :] < 0.5) & (deformed_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[1, :, :][(deformed_masks[0][b, 0, :, :] < 0.5) & (deformed_masks[1][b, 0, :, :] < 0.5) & (deformed_masks[2][b, 0, :, :] < 0.5)] = 1
        displace_mask[2, :, :][(deformed_masks[0][b, 0, :, :] < 0.5) & (deformed_masks[1][b, 0, :, :] < 0.5) & (deformed_masks[2][b, 0, :, :] < 0.5)] = 1

        displace_mask = displace_mask.permute(1, 2, 0)
        plt.imshow(displace_mask.detach().cpu().numpy())
        plot_grid(distx, disty, ax=ax, color="C0")

        fig, ax = plt.subplots()
        plot_grid(grid_x, grid_y, ax=ax, color="lightgrey")
        plt.imshow(image[b].squeeze(0).detach().cpu().numpy())
        plot_grid(distx, disty, ax=ax, color="C0")
        plt.show()


def train_shape_acdr_epoch(model, optimizer, data_loader, epoch, args, summary, device, affine_epoch):
    model.train()
    iterator = tqdm(data_loader)
    flow_grad_loss = Grad(penalty="l2")

    for i, (image, mask) in enumerate(iterator):

        image = image.to(device)
        mask = mask.to(device)

        # pred_masks, pred_nodes, disp, bdry, dt = model(image, mask, args.iter)
        # flow, preint_flow, init_mask0, init_mask1, init_mask2, \
        # affine_mask0, affine_mask1, affine_mask2, \
        # deform_mask0, deform_mask1, deform_mask2 = model(image, epoch)

        flow, preint_flow, init_mask0, init_mask1, init_mask2, \
        affine_mask0, affine_mask1, affine_mask2, \
        deform_mask0, deform_mask1, deform_mask2, out_par1, out_par2 = model(image, epoch, i == 0 and epoch % 30 == 0)
        if epoch % 30 == 0 and i == 0:
            print("init param", out_par1, out_par2)
        # plot_2(flow, image, mask, [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2])

        # g_map_lambda = 10
        # loss_gmap = g_map_lambda * F.mse_loss(disp, g_map)

        # loss_ac = loss_gmap
        loss_ac = 0
        global_step = epoch * len(data_loader) + i
        if epoch > affine_epoch:
            flow_loss = flow_grad_loss(flow)
            loss_ac += flow_loss
        m_f1 = []
        m_mse = []
        for mask_idx, affine_mask, init_mask, deform_mask in zip(
                [0, 1, 2],
                [affine_mask0, affine_mask1, affine_mask2],
                [init_mask0, init_mask1, init_mask2],
                [deform_mask0, deform_mask1, deform_mask2]
        ):
            loss_init_mask = F.mse_loss(init_mask.squeeze(1), mask[:, mask_idx, :, :], reduction="sum")
            loss_affine_mask = F.mse_loss(affine_mask.squeeze(1), mask[:, mask_idx, :, :], reduction="sum")
            loss_deform_mask = F.mse_loss(deform_mask.squeeze(1), mask[:, mask_idx, :, :], reduction="sum")

            # if epoch > affine_epoch:
            #     loss_ac += loss_init_mask + loss_affine_mask + loss_deform_mask
            # else:
            #     loss_ac += loss_init_mask + loss_affine_mask
            loss_ac += loss_init_mask

            f1_init_ac = np.mean(get_f1_scores(init_mask.gt(0.5), mask[:, mask_idx, :, :]))
            f1_affine_ac = np.mean(get_f1_scores(affine_mask.gt(0.5), mask[:, mask_idx, :, :]))
            f1_deform_ac = np.mean(get_f1_scores(deform_mask.gt(0.5), mask[:, mask_idx, :, :]))

            summary.add_scalar('train/f1_init_{}'.format(mask_idx), np.mean(f1_init_ac), global_step)
            summary.add_scalar('train/f1_affine_{}'.format(mask_idx), np.mean(f1_affine_ac), global_step)
            summary.add_scalar('train/f1_deform_{}'.format(mask_idx), np.mean(f1_deform_ac), global_step)
            summary.add_scalar('train/mse_deform_{}'.format(mask_idx), loss_deform_mask.item(), global_step)
            m_f1 += [f1_deform_ac]
            m_mse += [loss_deform_mask.item()]
        loss = loss_ac
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #
        iterator.set_description(
            '(train | {}) Epoch [{epoch}/{epochs}] :: Loss {loss:.4f} | Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss=loss.item(),
                loss_ac=loss_ac.item()))
        #
        global_step = epoch * len(data_loader) + i
        summary.add_scalar('train/loss', loss.item(), global_step)
        summary.add_scalar('train/m_f1_deform', np.mean(np.array(m_f1)), global_step)
        summary.add_scalar('train/m_mse_deform', np.mean(np.array(m_mse)), global_step)
        monai.visualize.img2tensorboard.plot_2d_or_3d_image(image, global_step, summary.writer, index=0, max_channels=1, max_frames=64,
                                                            tag='3d_image')
        monai.visualize.img2tensorboard.plot_2d_or_3d_image(mask, global_step, summary.writer, index=0, max_channels=1, max_frames=64,
                                                            tag='3d_mask')
        monai.visualize.img2tensorboard.plot_2d_or_3d_image(init_mask2, global_step, summary.writer, index=0, max_channels=1, max_frames=64,
                                                            tag='3d_init2')
        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[deform_mask0, deform_mask1, deform_mask2],
            affine_masks=[affine_mask0, affine_mask1, affine_mask2],
            init_masks=[init_mask0, init_mask1, init_mask2],
            global_step=global_step,
            prefix="train",
            output_dir=Path(summary.directory).joinpath("visualise_images")
        )



def train_shape_acdr_lv_epoch(model, optimizer, data_loader, epoch, args, summary, device, affine_epoch):
    model.train()
    iterator = tqdm(data_loader)
    flow_grad_loss = Grad(penalty="l2")
    for i, (image, mask) in enumerate(iterator):

        image = image.to(device)
        mask = mask.to(device)

        init_mask0, nodes0, out_par1 = model(image, epoch, i == 0)

        loss_ac = 0
        global_step = epoch * len(data_loader) + i

        m_f1 = []
        m_mse = []
        for mask_idx, init_mask in zip(
                [0],
                [init_mask0],
        ):
            loss_init_mask = F.mse_loss(init_mask.squeeze(1), mask[:, mask_idx, :, :], reduction='sum')
            loss_ac += loss_init_mask

            f1_init_ac = np.mean(get_f1_scores(init_mask.gt(0.5), mask[:, mask_idx, :, :]))

            summary.add_scalar('train/f1_init_{}'.format(mask_idx), np.mean(f1_init_ac), global_step)
            m_f1 += [f1_init_ac]
            m_mse += [loss_init_mask.item()]
        loss = loss_ac
        optimizer.zero_grad()
        loss.backward()
        print(i, nodes0.grad[0])
        print(out_par1[0])
        print()
        optimizer.step()

        #
        iterator.set_description(
            '(train | {}) Epoch [{epoch}/{epochs}] :: Loss {loss:.4f} | Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss=loss.item(),
                loss_ac=loss_ac.item()))
        #
        global_step = epoch * len(data_loader) + i
        summary.add_scalar('train/loss', loss.item(), global_step)
        summary.add_scalar('train/m_f1_deform', np.mean(np.array(m_f1)), global_step)
        summary.add_scalar('train/m_mse_deform', np.mean(np.array(m_mse)), global_step)
        monai.visualize.img2tensorboard.plot_2d_or_3d_image(image, global_step, summary.writer, index=0, max_channels=1, max_frames=64,
                                                            tag='3d_image')
        monai.visualize.img2tensorboard.plot_2d_or_3d_image(mask, global_step, summary.writer, index=0, max_channels=1, max_frames=64,
                                                            tag='3d_mask')
        monai.visualize.img2tensorboard.plot_2d_or_3d_image(init_mask0, global_step, summary.writer, index=0, max_channels=1, max_frames=64,
                                                            tag='3d_init2')
        visualise_images(
            image=image,
            mask=mask,
            pred_masks=None,
            affine_masks=None,
            init_masks=[init_mask0],
            global_step=global_step,
            prefix="train",
            output_dir=Path(summary.directory).joinpath("visualise_images")
        )


def val_shape_acdr_epoch(model, data_loader, epoch, args, summary, device):
    model.eval()
    iterator = tqdm(data_loader)

    mf1_init_ac = [[], [], []]
    mf1_affine_ac = [[], [], []]
    mf1_deform_ac = [[], [], []]

    mMask_ac = [[], [], []]
    m_loss = []
    for i, (image, mask) in enumerate(iterator):
        global_step = (epoch // args.eval_rate) * len(data_loader) + i
        image = image.to(device)
        mask = mask.to(device)

        flow, init_mask0, init_mask1, init_mask2, \
        affine_mask0, affine_mask1, affine_mask2, \
        deform_mask0, deform_mask1, deform_mask2 = model(image)

        # plot_2(flow, image, mask, [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2])

        # plot(flow, image, mask, [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2])

        for mask_idx, affine_mask, init_mask, deform_mask in zip(
            [0, 1, 2],
            [affine_mask0, affine_mask1, affine_mask2],
            [init_mask0, init_mask1, init_mask2],
            [deform_mask0, deform_mask1, deform_mask2],
        ):
            affine_mask_ac = F.interpolate(affine_mask, size=mask.shape[2:], mode='trilinear')
            affine_mask_ac = F.mse_loss(affine_mask_ac.squeeze(1), mask[:, mask_idx, :, :])

            init_mask_ac = F.interpolate(init_mask, size=mask.shape[2:], mode='trilinear')
            init_mask_ac = F.mse_loss(init_mask_ac.squeeze(1), mask[:, mask_idx, :, :])

            deform_mask_ac = F.interpolate(deform_mask, size=mask.shape[2:], mode='trilinear')
            deform_mask_ac = F.mse_loss(deform_mask_ac.squeeze(1), mask[:, mask_idx, :, :])

            # Metrices
            f1_init_ac = get_f1_scores(init_mask.gt(0.5), mask[:, mask_idx, :, :])
            f1_affine_ac = get_f1_scores(affine_mask.gt(0.5), mask[:, mask_idx, :, :])
            f1_deform_ac = get_f1_scores(deform_mask.gt(0.5), mask[:, mask_idx, :, :])
            mf1_init_ac[mask_idx] += f1_init_ac
            mf1_affine_ac[mask_idx] += f1_affine_ac
            mf1_deform_ac[mask_idx] += f1_deform_ac

            mMask_ac[mask_idx].append(deform_mask_ac.item())
            m_loss.append((deform_mask_ac + init_mask_ac + affine_mask_ac).item())
        # visualise_images(
        #     image=image,
        #     mask=mask,
        #     pred_masks=[deform_mask0, deform_mask1, deform_mask2],
        #     affine_masks=[affine_mask0, affine_mask1, affine_mask2],
        #     init_masks=[init_mask0, init_mask1, init_mask2],
        #     disp=None,
        #     g_map=g_map,
        #     global_step=global_step,
        #     prefix="val"
        # )

        iterator.set_description(
            '(val   | {}) Epoch [{epoch}/{epochs}] :: Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss_ac=np.mean(m_loss)))

    global_step = epoch // args.eval_rate
    for i in range(3):
        summary.add_scalar('val/m_f1_init_ac_{}'.format(i), np.mean(mf1_init_ac[i]), global_step)
        summary.add_scalar('val/m_f1_affine_ac_{}'.format(i), np.mean(mf1_affine_ac[i]), global_step)
        summary.add_scalar('val/m_f1_deform_ac_{}'.format(i), np.mean(mf1_deform_ac[i]), global_step)
        summary.add_scalar('val/m_mse_deform_ac_{}'.format(i), np.mean(mMask_ac[i]), global_step)

    global_step = epoch // args.eval_rate
    summary.add_scalar('val/m_f1_deform_ac', np.mean(np.array(mf1_deform_ac)), global_step)
    summary.add_scalar('val/m_mse_deform_ac',  np.mean(np.array(mMask_ac)), global_step)
    summary.add_scalar('val/loss_ac', np.mean(m_loss), global_step)
    return np.mean(np.array(mf1_deform_ac))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a segmentation')
    # Training
    parser.add_argument('--epochs', type=int,
                        default=5000,
                        help='Number of epochs')
    parser.add_argument('--start-epoch', type=int,
                        default=0,
                        help='Starting epoch')
    parser.add_argument('--batch-size', type=int,
                        default=5,
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-step', type=int,
                        default=200,
                        help='LR scheduler step')
    parser.add_argument("--affine-epoch", type=int, default=100, help="number of epochs with just affine loss")

    # Architecture
    parser.add_argument('--arch', type=str,
                        default='shape3dnet',
                        choices=["shape3dnet"],
                        help='Network architecture.')
    parser.add_argument('--voxel-width', type=int,
                        default=32,
                        help='Neural Renderer output size')
    parser.add_argument('--voxel-height', type=int,
                        default=32,
                        help='Neural Renderer output size')
    parser.add_argument('--voxel-depth', type=int,
                        default=32,
                        help='Neural Renderer output size')
    parser.add_argument('--enc-dim', type=int,
                        default=512,
                        help='Encoder dim(channels). Only relevant for UNet')

    # Buildings Data
    parser.add_argument('--data-path', type=str,
                        default='/path/to/building-dataset',
                        help='Path to buildings dataset directory')
    parser.add_argument('--template-path', type=str,
                        default='/path/to/building-dataset',
                        help='Path to buildings dataset directory')

    # Misc
    parser.add_argument('--eval-rate', type=int,
                        default=1,
                        help='Evaluate after eval_rate epochs')
    parser.add_argument('--save-rate', type=int,
                        default=1,
                        help='Save rate is save_rate * eval_rate')
    parser.add_argument('--checkname', type=str,
                        default='DEBUG',
                        help='Checkname')
    parser.add_argument('--resume', type=str,
                        default=None,
                        help='Resume file path')
    args = parser.parse_args()

    args.checkname = args.checkname + '_' + args.arch

    torch.multiprocessing.set_start_method('spawn')

    # Define Saver

    saver = Saver(args)
    saver.save_experiment_config()

    # Define Tensorboard Summary
    summary = TensorboardSummary(saver.experiment_dir)
    args.exp = saver.experiment_dir.split('_')[-1]

    # Data
    train_ds = Cardiac3dDataset(Path(args.data_path), args.voxel_width, args.voxel_height, args.voxel_depth, Path(saver.experiment_dir))
    val_ds = Cardiac3dDataset(Path(args.data_path),  args.voxel_width, args.voxel_height, args.voxel_depth, Path(saver.experiment_dir))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ShapeDeformCardiac3DCircleNet(
        num_nodes=args.num_nodes,
        enc_dim=args.enc_dim,
        voxel_width=args.voxel_width,
        voxel_height=args.voxel_height,
        voxel_depth=args.voxel_depth,
        drop=args.drop,
        num_lv_slices=64,
        num_extra_lv_myo_slices=16
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, gamma=0.1)

    if args.resume:
        if args.resume == 'last':
            args.resume = os.path.join(saver.directory, 'model_last.pth.tar')
        elif args.resume == 'best':
            args.resume = os.path.join(saver.directory, 'model_best.pth.tar')
        if not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler']),
        best_pred = checkpoint['best_pred']

    best_pred = 0
    for epoch in range(args.start_epoch, args.epochs):
        # scheduler.step()
        with torch.autograd.set_detect_anomaly(True):
            train_shape_acdr_epoch(model, optimizer, train_dl, epoch, args, summary, device, args.affine_epoch)
            # train_shape_acdr_lv_epoch(model, optimizer, train_dl, epoch, args, summary, device, args.affine_epoch)

        # if epoch % args.eval_rate == 0:
        #     mIoU_ac = val_shape_acdr_epoch(model, val_dl, epoch, args, summary, device)
        #
        #     is_best = False
        #     if mIoU_ac > best_pred:
        #         best_pred = mIoU_ac
        #         is_best = True
        #
        #     if epoch % (args.save_rate * args.eval_rate) == 0:
        #         model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        #         saver.save_checkpoint({
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             # 'scheduler': scheduler.state_dict(),
        #             'best_pred': best_pred,
        #         }, is_best)
