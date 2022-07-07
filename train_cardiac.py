import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

from models.networks.cardiac import CardaicCircleNet, CardiacUNet
from models.networks.deform import DeformCardiacCircleNet
from models.networks.shape import ShapeDeformCardiacCircleNet
# from models.networks.shape_2 import ShapeDeformCardiacCircleNet

from sklearn.neural_network import BernoulliRBM

from data.cardiac import CardiacDataset
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrices import get_ap_scores, get_iou, get_f1_scores
from data import transforms
from models.loss_functions import curvature_loss, dist_loss
from torchvision.utils import make_grid
from models.loss_functions import Grad
# --resume
# C:\Users\lisur\PycharmProjects\ACDRNet-master\run\buildings\rider_DEBUG_unet\experiment_16\checkpoint.pth.tar


def visualise_images(image, mask, pred_masks, affine_masks, disp, g_map, global_step, prefix: str = "train", init_masks=None):
    grid_image = make_grid(image[:10, :, :, :].clone().cpu().data, 10, normalize=True, scale_each=True)
    summary.writer.add_image(f'{prefix}_image/image', grid_image, global_step)
    displace_mask = torch.zeros_like(mask[:10, :, :, :])
    mask[:, 1, :, :][mask[:, 0, :, :] > 0.5] = 0
    displace_mask[:, 0, :, :][mask[:10, 0, :, :] > 0.5] = 1
    displace_mask[:, 1, :, :][mask[:10, 1, :, :] > 0.5] = 1
    displace_mask[:, 2, :, :][mask[:10, 2, :, :] > 0.5] = 1
    grid_image = make_grid(displace_mask.clone().cpu().data, 10, normalize=True, scale_each=True)
    summary.writer.add_image(f'{prefix}_mask/mask', grid_image, global_step)

    if pred_masks is not None:
        displace_mask = torch.zeros_like(mask[:10, :, :, :])
        pred_masks[1][pred_masks[0] > 0.5] = 0
        for idx in range(len(pred_masks)):
            displace_mask[:, idx:idx+1, :, :][pred_masks[idx][:10] > 0.5] = 1
        grid_image = make_grid(displace_mask.clone().cpu().data, 10, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}_pred/pred', grid_image, global_step)

    if affine_masks is not None:
        affine_masks[1][affine_masks[0] > 0.5] = 0
        displace_mask = torch.zeros_like(mask[:10, :, :, :])
        displace_mask[:, 0:1, :, :][affine_masks[0][:10] > 0.5] = 1
        displace_mask[:, 1:2, :, :][affine_masks[1][:10] > 0.5] = 1
        displace_mask[:, 2:3, :, :][affine_masks[2][:10] > 0.5] = 1
        grid_image = make_grid(displace_mask.clone().cpu().data, 10, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}_pred/affine', grid_image, global_step)

    if init_masks is not None:
        init_masks[1][init_masks[0] > 0.5] = 0
        displace_mask = torch.zeros_like(mask[:10, :, :, :])
        displace_mask[:, 0:1, :, :][init_masks[0][:10] > 0.5] = 1
        displace_mask[:, 1:2, :, :][init_masks[1][:10] > 0.5] = 1
        displace_mask[:, 2:3, :, :][init_masks[2][:10] > 0.5] = 1
        grid_image = make_grid(displace_mask.clone().cpu().data, 10, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}_pred/init', grid_image, global_step)


    if disp is not None:
        grid_image = make_grid(disp[:3, 0:1, :, :].clone().cpu().data, 3, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}/Disp0_x', grid_image, global_step)
        grid_image = make_grid(disp[:3, 1:2, :, :].clone().cpu().data, 3, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}/Disp0_y', grid_image, global_step)

    if g_map is not None:
        grid_image = make_grid(g_map[:3, 0:1, :, :].clone().cpu().data, 3, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}/Gmap0_x', grid_image, global_step)

        grid_image = make_grid(g_map[:3, 1:2, :, :].clone().cpu().data, 3, normalize=True, scale_each=True)
        summary.writer.add_image(f'{prefix}/Gmap0_y', grid_image, global_step)


def train_segnet_epoch(model, optimizer, data_loader, epoch, args, summary, device):
    model.train()
    iterator = tqdm(data_loader)

    for i, (image, mask, g_map) in enumerate(iterator):

        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)
        global_step = epoch * len(data_loader) + i
        loss = 0

        # pred_masks, pred_nodes, disp, bdry, dt = model(image, mask, args.iter)
        pred_masks = model(image)
        for mask_idx in [1, 2]:

            loss_masks = F.mse_loss(pred_masks[:, mask_idx-1, :, :], mask[:, mask_idx, :, :])
            loss += loss_masks

            ap_ac = np.mean(get_ap_scores(pred_masks[:, mask_idx-1, :, :].gt(0.5), mask[:, mask_idx, :, :]))
            f1_ac = np.mean(get_f1_scores(pred_masks[:, mask_idx-1, :, :].gt(0.5), mask[:, mask_idx, :, :]))
            iou_ac = np.mean(get_iou(pred_masks[:, mask_idx-1, :, :].gt(0.5), mask[:, mask_idx, :, :].bool()))

            summary.add_scalar('train/loss_masks_{}'.format(mask_idx), loss_masks.item(), global_step)
            summary.add_scalar('train/iou_ac_{}'.format(mask_idx), np.mean(iou_ac), global_step)
            summary.add_scalar('train/ap_ac_{}'.format(mask_idx), np.mean(ap_ac), global_step)
            summary.add_scalar('train/f1_ac_{}'.format(mask_idx), np.mean(f1_ac), global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #
        iterator.set_description(
            '(train | {}) Epoch [{epoch}/{epochs}] :: Loss {loss:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss=loss.item()),
        )
        #
        global_step = epoch * len(data_loader) + i
        summary.add_scalar('train/loss', loss.item(), global_step)

        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[pred_masks[:, 0:1, :, :], pred_masks[:, 1:2, :, :]],
            affine_masks=None,
            disp=None,
            g_map=None,
            global_step=global_step,
            prefix="train"
        )


def val_segnet_epoch(model, data_loader, epoch, args, summary, device):
    model.eval()
    iterator = tqdm(data_loader)

    mIoU_ac, mAP_ac, mF1_ac = [[], []], [[], []], [[], []]
    mMask_ac = [[], []]
    for i, (image, mask, g_map) in enumerate(iterator):
        global_step = (epoch // args.eval_rate) * len(data_loader) + i
        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        pred_masks = model(image)
        loss_masks_ac_total = 0
        for mask_idx in [1, 2]:

            loss_masks_ac = F.mse_loss(pred_masks[:, mask_idx-1, :, :].squeeze(1), mask[:, mask_idx, :, :])

            # Metrices
            iou_ac = get_iou(pred_masks[:, mask_idx-1, :, :].gt(0.5), mask[:, mask_idx, :, :].bool())
            ap_ac = get_ap_scores(pred_masks[:, mask_idx-1, :, :].gt(0.5), mask[:, mask_idx, :, :])
            f1_ac = get_f1_scores(pred_masks[:, mask_idx-1, :, :].gt(0.5), mask[:, mask_idx, :, :])
            mIoU_ac[mask_idx - 1] += iou_ac
            mAP_ac[mask_idx - 1] += ap_ac
            mF1_ac[mask_idx - 1] += f1_ac
            mMask_ac[mask_idx - 1] += [loss_masks_ac.item()]
            loss_masks_ac_total += loss_masks_ac

            ind = np.argwhere(np.array(iou_ac) < 0.5).flatten().tolist()
            if ind:
                summary.visualize_image(
                    'val_BAD_{}'.format(mask_idx),
                    image[ind],
                    mask[:, mask_idx, :, :][ind].unsqueeze(1),
                    pred_masks[:, mask_idx-1, :, :].unsqueeze(1)[ind],
                    pred_masks[:, mask_idx-1, :, :].unsqueeze(1)[ind],
                    global_step
                )

        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[pred_masks[:, 0:1, :, :], pred_masks[:, 1:2, :, :]],
            affine_masks=None,
            disp=None,
            g_map=None,
            global_step=global_step,
            prefix="val"
        )

        iterator.set_description(
            '(val   | {}) Epoch [{epoch}/{epochs}] :: Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss_ac=loss_masks_ac_total.item()))

    global_step = epoch // args.eval_rate
    for i in range(2):
        summary.add_scalar('val/mIoU_ac_{}'.format(i+1), np.mean(mIoU_ac[i]), global_step)
        summary.add_scalar('val/mAP_ac_{}'.format(i+1), np.mean(mAP_ac[i]), global_step)
        summary.add_scalar('val/mF1_ac_{}'.format(i+1), np.mean(mF1_ac[i]), global_step)
        summary.add_scalar('val/mMask_ac_{}'.format(i+1), np.mean(mMask_ac[i]), global_step)

    return np.mean(np.array(mIoU_ac)), np.mean(np.array(mAP_ac)), np.mean(np.array(mF1_ac)), np.mean(np.array(mMask_ac))


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
    print(grid.shape)
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


def train_acdr_epoch(model, optimizer, data_loader, epoch, args, summary, device, affine_epoch):
    model.train()
    iterator = tqdm(data_loader)
    flow_grad_loss = Grad(penalty="l2")

    for i, (image, mask, g_map) in enumerate(iterator):

        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        # pred_masks, pred_nodes, disp, bdry, dt = model(image, mask, args.iter)
        pred_masks0, pred_masks1, pred_masks2, nodes0, nodes1, nodes2, disp, preint_flow = model(image, args.iter)
        plot(disp, image, mask, [pred_masks0[0], pred_masks1[0], pred_masks2[0]], [pred_masks0[-1], pred_masks1[-1], pred_masks2[-1]])
        start_index = 1
        # g_map_lambda = 10
        # loss_gmap = g_map_lambda * F.mse_loss(disp, g_map)

        # loss_ac = loss_gmap
        loss_ac = 0
        global_step = epoch * len(data_loader) + i
        if epoch > affine_epoch:
            flow_loss = flow_grad_loss(preint_flow)
            loss_ac += flow_loss * 10000

        for mask_idx, pred_masks, nodes in zip([0, 1, 2], [pred_masks0, pred_masks1, pred_masks2], [nodes0, nodes1, nodes2]):
            loss_masks = [
                F.mse_loss(pred_masks[k].squeeze(1), mask[:, mask_idx, :, :])
                for k in range(start_index, len(pred_masks))
            ]
            loss_affine_masks = F.mse_loss(pred_masks[0].squeeze(1), mask[:, mask_idx, :, :])
            loss_balloon = [(1 - pred_masks[k]).mean() for k in range(len(pred_masks))]
            loss_curve = [curvature_loss(verts) for verts in nodes]

            loss_masks_agg = []
            loss_balloon_agg = []
            loss_curve_agg = []

            loss_masks_agg.append(loss_masks[-1])

            loss_balloon_agg.append(args.lmd_balloon * loss_balloon[-1])
            loss_curve_agg.append(args.lmd_curve * loss_curve[-1])

            if len(loss_masks) > 2:
                loss_masks_agg += [loss_masks[j + start_index] for j in range(len(loss_masks[start_index:-1]))]

                loss_balloon_agg += [
                    args.lmd_balloon * loss_balloon[j + start_index]
                    for j in range(len(loss_masks[start_index:-1]))
                ]
                loss_curve_agg += [
                    args.lmd_curve * loss_curve[j + start_index]
                    for j in range(len(loss_masks[start_index:-1]))
                ]
            if epoch > affine_epoch:
                loss_ac += sum(loss_masks_agg) + loss_affine_masks
            else:
                loss_ac += loss_affine_masks

            iou_ac = np.mean(get_iou(pred_masks[-1].gt(0.5), mask[:, mask_idx, :, :].bool()))
            affine_iou_ac = np.mean(get_iou(pred_masks[0].gt(0.5), mask[:, mask_idx, :, :].bool()))

            ap_ac = np.mean(get_ap_scores(pred_masks[-1].gt(0.5), mask[:, mask_idx, :, :]))
            f1_ac = np.mean(get_f1_scores(pred_masks[-1].gt(0.5), mask[:, mask_idx, :, :]))

            summary.add_scalar('train/loss_masks_agg_{}'.format(mask_idx), sum(loss_masks_agg).item(), global_step)

            summary.add_scalar('train/loss_ballon_agg_{}'.format(mask_idx), sum(loss_balloon_agg).item(), global_step)
            summary.add_scalar('train/loss_curv_agg_{}'.format(mask_idx), sum(loss_curve_agg).item(), global_step)

            summary.add_scalar('train/iou_ac_{}'.format(mask_idx), np.mean(iou_ac), global_step)
            summary.add_scalar('train/affine_iou_ac_{}'.format(mask_idx), np.mean(affine_iou_ac), global_step)

            summary.add_scalar('train/ap_ac_{}'.format(mask_idx), np.mean(ap_ac), global_step)
            summary.add_scalar('train/f1_ac_{}'.format(mask_idx), np.mean(f1_ac), global_step)

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
        summary.add_scalar('train/loss_ac', loss_ac.item(), global_step)

        # summary.add_scalar('train/loss_dist_agg', sum(loss_dist_agg).item(), global_step)
        # summary.add_scalar('train/loss_gmap', loss_gmap.item(), global_step)
        #

        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[pred_masks0[-1], pred_masks1[-1], pred_masks2[-1]],
            affine_masks=[pred_masks0[0], pred_masks1[0], pred_masks2[0]],
            disp=disp,
            g_map=g_map,
            global_step=global_step,
            prefix="train"
        )


def val_acdr_epoch(model, data_loader, epoch, args, summary, device):
    model.eval()
    iterator = tqdm(data_loader)

    mIoU_ac, mAP_ac, mF1_ac = [[], [], []], [[], [], []], [[], [], []]
    mMask_ac = [[], [], []]
    for i, (image, mask, g_map) in enumerate(iterator):
        global_step = (epoch // args.eval_rate) * len(data_loader) + i
        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        affine_mask0, pred_mask0, affine_mask1, pred_mask1, affine_mask2, pred_mask2, disp = model(image, args.iter)

        # plot(disp, image, mask, [affine_mask0, affine_mask1, affine_mask2], [pred_mask0, pred_mask1, pred_mask2])

        loss_masks_ac_total = 0
        for mask_idx, pred_mask in zip([0, 1, 2], [pred_mask0, pred_mask1, pred_mask2]):
            pred_mask_ac = F.interpolate(pred_mask, size=mask.shape[2:], mode='bilinear')

            loss_masks_ac = F.mse_loss(pred_mask_ac.squeeze(1), mask[:, mask_idx, :, :])

            # Metrices
            iou_ac = get_iou(pred_mask_ac.gt(0.5), mask[:, mask_idx, :, :].bool())
            ap_ac = get_ap_scores(pred_mask_ac.gt(0.5), mask[:, mask_idx, :, :])
            f1_ac = get_f1_scores(pred_mask_ac.gt(0.5), mask[:, mask_idx, :, :])
            mIoU_ac[mask_idx] += iou_ac
            mAP_ac[mask_idx] += ap_ac
            mF1_ac[mask_idx] += f1_ac
            mMask_ac[mask_idx] += [loss_masks_ac.item()]
            loss_masks_ac_total += loss_masks_ac

            ind = np.argwhere(np.array(iou_ac) < 0.5).flatten().tolist()
            if ind:
                summary.visualize_image(
                    'val_BAD_{}'.format(mask_idx),
                    image[ind],
                    mask[:, mask_idx, :, :][ind].unsqueeze(1),
                    pred_mask_ac[ind],
                    pred_mask_ac[ind],
                    global_step
                )
                # grid_image = make_grid(
                #     disp[ind][:3, (mask_idx) * 2, :, :].unsqueeze(1).clone().cpu().data, 3,
                #     normalize=True, scale_each=True
                # )
                # summary.writer.add_image('val_BAD_{}/Disp_x'.format(mask_idx), grid_image, global_step)

                # grid_image = make_grid(
                #     disp[ind][:3, (mask_idx) * 2 + 1, :, :].unsqueeze(1).clone().cpu().data, 3,
                #     normalize=True, scale_each=True
                # )
                # summary.writer.add_image('val_BAD_{}/Disp_y'.format(mask_idx), grid_image, global_step)

                grid_image = make_grid(
                    g_map[ind][:3, (mask_idx) * 2, :, :].unsqueeze(1).clone().cpu().data, 3,
                    normalize=True, scale_each=True
                )
                summary.writer.add_image('val_BAD_{}/Gmap_x'.format(mask_idx), grid_image, global_step)

                grid_image = make_grid(
                    g_map[ind][:3, (mask_idx) * 2 + 1, :, :].unsqueeze(1).clone().cpu().data, 3,
                    normalize=True, scale_each=True
                )
                summary.writer.add_image('val_BAD_{}/Gmap_y'.format(mask_idx), grid_image, global_step)
        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[pred_mask0, pred_mask1, pred_mask2],
            affine_masks=[affine_mask0, affine_mask1, affine_mask2],
            disp=disp,
            g_map=g_map,
            global_step=global_step,
            prefix="val"
        )

        iterator.set_description(
            '(val   | {}) Epoch [{epoch}/{epochs}] :: Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss_ac=loss_masks_ac_total.item()))

        # summary.add_scalar('val/loss_ac', loss_masks_ac.item(), global_step)
        # summary.add_scalar('val/iou_ac', np.mean(iou_ac), global_step)
        # summary.add_scalar('val/ap_ac', np.mean(ap_ac), global_step)

    global_step = epoch // args.eval_rate
    for i in range(3):
        summary.add_scalar('val/mIoU_ac_{}'.format(i), np.mean(mIoU_ac[i]), global_step)
        summary.add_scalar('val/mAP_ac_{}'.format(i), np.mean(mAP_ac[i]), global_step)
        summary.add_scalar('val/mF1_ac_{}'.format(i), np.mean(mF1_ac[i]), global_step)
        summary.add_scalar('val/mMask_ac_{}'.format(i), np.mean(mMask_ac[i]), global_step)

    return np.mean(np.array(mIoU_ac)), np.mean(np.array(mAP_ac)), np.mean(np.array(mF1_ac)), np.mean(np.array(mMask_ac))


def train_shape_acdr_epoch(model, optimizer, data_loader, epoch, args, summary, device, affine_epoch):
    model.train()
    iterator = tqdm(data_loader)
    flow_grad_loss = Grad(penalty="l2")

    for i, (image, mask, g_map) in enumerate(iterator):

        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        # pred_masks, pred_nodes, disp, bdry, dt = model(image, mask, args.iter)
        flow, preint_flow, init_mask0, init_mask1, init_mask2, \
        affine_mask0, affine_mask1, affine_mask2, \
        deform_mask0, deform_mask1, deform_mask2 = model(image, args.iter)

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
            loss_init_mask = F.mse_loss(init_mask.squeeze(1), mask[:, mask_idx, :, :])
            loss_affine_mask = F.mse_loss(affine_mask.squeeze(1), mask[:, mask_idx, :, :])
            loss_deform_mask = F.mse_loss(deform_mask.squeeze(1), mask[:, mask_idx, :, :])

            if epoch > affine_epoch:
                loss_ac += loss_init_mask + loss_affine_mask + loss_deform_mask
            else:
                loss_ac += loss_init_mask + loss_affine_mask

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

        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[deform_mask0, deform_mask1, deform_mask2],
            affine_masks=[affine_mask0, affine_mask1, affine_mask2],
            init_masks=[init_mask0, init_mask1, init_mask2],
            disp=None,
            g_map=g_map,
            global_step=global_step,
            prefix="train"
        )


def val_shape_acdr_epoch(model, data_loader, epoch, args, summary, device):
    model.eval()
    iterator = tqdm(data_loader)

    mf1_init_ac = [[], [], []]
    mf1_affine_ac = [[], [], []]
    mf1_deform_ac = [[], [], []]

    mMask_ac = [[], [], []]
    m_loss = []
    for i, (image, mask, g_map) in enumerate(iterator):
        global_step = (epoch // args.eval_rate) * len(data_loader) + i
        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        flow, init_mask0, init_mask1, init_mask2, \
        affine_mask0, affine_mask1, affine_mask2, \
        deform_mask0, deform_mask1, deform_mask2 = model(image)

        plot_2(flow, image, mask, [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2])

        # plot(flow, image, mask, [affine_mask0, affine_mask1, affine_mask2], [deform_mask0, deform_mask1, deform_mask2])

        for mask_idx, affine_mask, init_mask, deform_mask in zip(
            [0, 1, 2],
            [affine_mask0, affine_mask1, affine_mask2],
            [init_mask0, init_mask1, init_mask2],
            [deform_mask0, deform_mask1, deform_mask2],
        ):
            affine_mask_ac = F.interpolate(affine_mask, size=mask.shape[2:], mode='bilinear')
            affine_mask_ac = F.mse_loss(affine_mask_ac.squeeze(1), mask[:, mask_idx, :, :])

            init_mask_ac = F.interpolate(init_mask, size=mask.shape[2:], mode='bilinear')
            init_mask_ac = F.mse_loss(init_mask_ac.squeeze(1), mask[:, mask_idx, :, :])

            deform_mask_ac = F.interpolate(deform_mask, size=mask.shape[2:], mode='bilinear')
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
        visualise_images(
            image=image,
            mask=mask,
            pred_masks=[deform_mask0, deform_mask1, deform_mask2],
            affine_masks=[affine_mask0, affine_mask1, affine_mask2],
            init_masks=[init_mask0, init_mask1, init_mask2],
            disp=None,
            g_map=g_map,
            global_step=global_step,
            prefix="val"
        )

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
                        default=64,
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
                        default='acdrnet',
                        choices=['acdrnet', 'segnet', "deformnet", "shapenet"],
                        help='Network architecture. acdrnet or segnet')
    parser.add_argument('--image-size', type=int,
                        default=128,
                        help='Neural Renderer output size')
    parser.add_argument('--dec-size', type=int,
                        default=64,
                        help='Spatial size of the decoder. Only relevant for ResNet')
    parser.add_argument('--enc-dim', type=int,
                        default=512,
                        help='Encoder dim(channels). Only relevant for UNet')
    parser.add_argument('--dec-dim', type=int,
                        default=256,
                        help='Decoder dim(channels)')
    parser.add_argument('--drop', type=float,
                        default=0.1,
                        help='Dropout rate')

    # Active contour
    parser.add_argument('--num-nodes', type=int,
                        default=70,
                        help='Number of nodes')
    parser.add_argument('--iter', type=int,
                        default=3,
                        help='AC number of iterations')
    parser.add_argument('--lmd-balloon', type=float,
                        default=0.01,
                        help='Balloon')
    parser.add_argument('--lmd-curve', type=float,
                        default=0.001,
                        help='Curvature')
    parser.add_argument('--lmd-dist', type=float,
                        default=0.1,
                        help='Distance')

    # Data
    parser.add_argument('--train-dataset', type=str,
                        default='cityscapes',
                        help='Training dataset')
    parser.add_argument('--ann-train', type=str,
                        default='train',
                        help='Split for training')
    parser.add_argument('--ann-val', type=str,
                        default='val',
                        help='Split tor evaluation')

    # Buildings Data
    parser.add_argument('--data-path', type=str,
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
    MEAN = np.array([0.47341759 * 255, 0.28791303 * 255, 0.2850705 * 255])
    STD = np.array([0.22645572 * 255, 0.15276193 * 255, 0.140702 * 255])
    train_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.RandomAffineFromSet(degrees=[0, 15, 60, 90, 135, 180, 225, 270], scale=(0.75, 1.25)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        # transforms.NormalizeInstance(),
        # transforms.Normalize(MEAN, STD),
    ])
    val_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        # transforms.NormalizeInstance()
        # transforms.Normalize(MEAN, STD),
    ])

    train_ds = CardiacDataset(Path(args.data_path), args.ann_train, transformations=train_trans)
    val_ds = CardiacDataset(Path(args.data_path), args.ann_val, transformations=val_trans)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.arch == "acdrnet":
        model = CardaicCircleNet(
            num_nodes=args.num_nodes,
            enc_dim=args.enc_dim,
            dec_size=args.dec_size,
            image_size=args.image_size,
            drop=args.drop,
        ).to(device)
    elif args.arch == "segnet":
        model = CardiacUNet(
            enc_dim=args.enc_dim,
            drop=args.drop,
            output_dim=3
        ).to(device)
    elif args.arch == "deformnet":
        model = DeformCardiacCircleNet(
            num_nodes=args.num_nodes,
            enc_dim=args.enc_dim,
            dec_size=args.dec_size,
            image_size=args.image_size,
            drop=args.drop,
        ).to(device)
    elif args.arch == "shapenet":
        model = ShapeDeformCardiacCircleNet(
            num_nodes=args.num_nodes,
            enc_dim=args.enc_dim,
            dec_size=args.dec_size,
            image_size=args.image_size,
            drop=args.drop,
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
            if args.arch == "acdrnet" or args.arch == "deformnet":
                train_acdr_epoch(model, optimizer, train_dl, epoch, args, summary, device, args.affine_epoch)
            if args.arch == "segnet":
                train_segnet_epoch(model, optimizer, train_dl, epoch, args, summary, device)
            if args.arch == "shapenet":
                train_shape_acdr_epoch(model, optimizer, train_dl, epoch, args, summary, device, args.affine_epoch)

        if epoch % args.eval_rate == 0:
            if args.arch == "acdrnet" or args.arch == "deformnet":
                mIoU_ac, mAP_ac, mF1_ac, mMask_ac = val_acdr_epoch(model, val_dl, epoch, args, summary, device)
                global_step = epoch // args.eval_rate
                summary.add_scalar('val/mIoU_ac', mIoU_ac, global_step)
                summary.add_scalar('val/mAP_ac', mAP_ac, global_step)
                summary.add_scalar('val/mF1_ac', mF1_ac, global_step)
                summary.add_scalar('val/mMask_ac', mMask_ac, global_step)
            if args.arch == "segnet":
                mIoU_ac, mAP_ac, mF1_ac, mMask_ac = val_segnet_epoch(model, val_dl, epoch, args, summary, device)

                global_step = epoch // args.eval_rate
                summary.add_scalar('val/mIoU_ac', mIoU_ac, global_step)
                summary.add_scalar('val/mAP_ac', mAP_ac, global_step)
                summary.add_scalar('val/mF1_ac', mF1_ac, global_step)
                summary.add_scalar('val/mMask_ac', mMask_ac, global_step)

            if args.arch == "shapenet":
                mIoU_ac = val_shape_acdr_epoch(model, val_dl, epoch, args, summary, device)

            is_best = False
            if mIoU_ac > best_pred:
                best_pred = mIoU_ac
                is_best = True

            if epoch % (args.save_rate * args.eval_rate) == 0:
                model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    'best_pred': best_pred,
                }, is_best)


def plot():
    # nodes0, nodes2, mask0, mask2, face0, face2, a_mask0, a_mask2 = model(image, mask, args.iter)
    # global_step = epoch * len(data_loader) + i
    # from matplotlib import pyplot as plt
    # half_dim = 128/2
    # nodes0[:, :, :, 1] = -nodes0[:, :, :, 1]
    # nodes0 = nodes0 * half_dim + half_dim
    #
    # nodes2[:, :, :, 1] = -nodes2[:, :, :, 1]
    # nodes2 = nodes2 * half_dim + half_dim
    #
    # nodes0 = nodes0[0, 0, :, :].detach().cpu().numpy()
    # nodes2 = nodes2[0, 0, :, :].detach().cpu().numpy()
    # mask0 = mask0[0, 0, :, :].detach().cpu().numpy()
    # mask2 = mask2[0, 0, :, :].detach().cpu().numpy()
    # a_mask0 = a_mask0[0, 0, :, :].detach().cpu().numpy()
    # a_mask2 = a_mask2[0, 0, :, :].detach().cpu().numpy()
    #
    # face0 = face0[0, 0, :, :].detach().cpu().numpy()
    # face2 = face2[0, 0, :, :].detach().cpu().numpy()
    # plt.figure()
    # plt.imshow(mask0)
    # plt.figure()
    # plt.imshow(mask2)
    # plt.figure()
    # plt.imshow(a_mask0)
    # plt.figure()
    # plt.imshow(a_mask2)
    # plt.show()
    #
    #
    # # plt.show()
    # # plt.figure()
    # # plt.imshow(np.zeros((128, 128)))
    # plt.plot(nodes2[0, 0], nodes2[0, 1], 'go')
    # plt.plot(nodes2[-1, 0], nodes2[-1, 1], 'ro')
    #
    # plt.plot(nodes2[:, 0], nodes2[:, 1], 'bx-')
    # for f in range(face2.shape[0]):
    #     p1 = nodes2[face2[f, 0], :]
    #     p2 = nodes2[face2[f, 1], :]
    #     p3 = nodes2[face2[f, 2], :]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    #     plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
    #     plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')
    #
    # plt.show()
    #
    # plt.figure()
    #
    # plt.plot(nodes0[0, 0], nodes0[0, 1], 'go')
    # plt.plot(nodes0[-1, 0], nodes0[-1, 1], 'ro')
    #
    # plt.plot(nodes0[:, 0], nodes0[:, 1], 'bx-')
    # for f in range(face0.shape[0]):
    #     p1 = nodes0[face0[f, 0], :]
    #     p2 = nodes0[face0[f, 1], :]
    #     p3 = nodes0[face0[f, 2], :]
    #     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    #     plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
    #     plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')
    #
    # plt.show()
    # summary.writer.add_image('train/mask0', mask0[0, :1, :, :], global_step)
    # summary.writer.add_image('train/mask1', mask1[0, :1, :, :], global_step)

    pass