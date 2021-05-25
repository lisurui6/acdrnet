import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

from models import networks
from data.cityscapes_instances import CityscapesInstances, CityscapesInstances_comp
from data.buildings import BuildingsDataset
from data.cardiac import CardiacDataset
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrices import get_ap_scores, get_iou, get_f1_scores
from data import transforms
from models.loss_functions import curvature_loss, dist_loss
from torchvision.utils import make_grid
# --resume
# C:\Users\lisur\PycharmProjects\ACDRNet-master\run\buildings\rider_DEBUG_unet\experiment_16\checkpoint.pth.tar


def train_epoch(model, optimizer, data_loader, epoch, args, summary, device):
    model.train()
    iterator = tqdm(data_loader)

    for i, (image, mask, g_map) in enumerate(iterator):

        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        pred_masks, pred_nodes, disp, bdry, dt = model(image, args.iter)

        start_index = 1
        g_map_lambda = 10

        loss_masks = [F.mse_loss(pred_masks[k].squeeze(1), mask) for k in range(start_index, len(pred_masks))]
        loss_balloon = [(1 - pred_masks[k]).mean() for k in range(len(pred_masks))]
        loss_curve = [curvature_loss(nodes) for nodes in pred_nodes]
        loss_dist = [dist_loss(nodes) for nodes in pred_nodes]

        loss_gmap = g_map_lambda * F.mse_loss(disp, g_map)

        loss_masks_agg = []
        loss_balloon_agg = []
        loss_curve_agg = []
        loss_dist_agg = []

        loss_masks_agg.append(loss_masks[-1])
        loss_balloon_agg.append(args.lmd_balloon * loss_balloon[-1])
        loss_curve_agg.append(args.lmd_curve * loss_curve[-1])
        loss_dist_agg.append(args.lmd_dist * loss_dist[-1])

        if len(loss_masks) > 2:
            loss_masks_agg += [loss_masks[j + start_index]
                               for j in range(len(loss_masks[start_index:-1]))]
            loss_balloon_agg += [args.lmd_balloon * loss_balloon[j + start_index]
                                 for j in range(len(loss_masks[start_index:-1]))]
            loss_curve_agg += [args.lmd_curve * loss_curve[j + start_index]
                               for j in range(len(loss_masks[start_index:-1]))]
            loss_dist_agg += [args.lmd_dist * loss_dist[j + start_index]
                              for j in range(len(loss_masks[start_index:-1]))]

        loss_ac = sum(loss_masks_agg) + sum(loss_balloon_agg) + sum(loss_dist_agg) + sum(loss_curve_agg) + loss_gmap
        loss = loss_ac

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrices
        iou_ac = np.mean(get_iou(pred_masks[-1].gt(0.5), mask.bool()))
        ap_ac = np.mean(get_ap_scores(pred_masks[-1].gt(0.5), mask))
        f1_ac = np.mean(get_f1_scores(pred_masks[-1].gt(0.5), mask))

        iterator.set_description(
            '(train | {}) Epoch [{epoch}/{epochs}] :: Loss {loss:.4f} | Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss=loss.item(),
                loss_ac=loss_ac.item()))

        global_step = epoch * len(data_loader) + i
        summary.add_scalar('train/loss', loss.item(), global_step)
        summary.add_scalar('train/loss_ac', loss_ac.item(), global_step)
        summary.add_scalar('train/loss_masks_agg', sum(loss_masks_agg).item(), global_step)
        summary.add_scalar('train/loss_ballon_agg', sum(loss_balloon_agg).item(), global_step)
        summary.add_scalar('train/loss_curv_agg', sum(loss_curve_agg).item(), global_step)
        summary.add_scalar('train/loss_dist_agg', sum(loss_dist_agg).item(), global_step)
        summary.add_scalar('train/loss_gmap', loss_gmap.item(), global_step)

        summary.add_scalar('train/iou_ac', np.mean(iou_ac), global_step)
        summary.add_scalar('train/ap_ac', np.mean(ap_ac), global_step)
        summary.add_scalar('train/f1_ac', np.mean(f1_ac), global_step)

        summary.visualize_image('train',
                                image,
                                mask.unsqueeze(1),
                                pred_masks[-1],
                                pred_masks[0],
                                global_step)
        grid_image = make_grid(disp[:3, :1, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('train/Disp_x', grid_image, global_step)
        grid_image = make_grid(disp[:3, 1:, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('train/Disp_y', grid_image, global_step)

        grid_image = make_grid(g_map[:3, :1, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('train/Gmap_x', grid_image, global_step)

        grid_image = make_grid(g_map[:3, 1:, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('train/Gmap_y', grid_image, global_step)

        for c in range(bdry.shape[1]):
            # grid_image = make_grid(bdry[:3, c, :, :].unsqueeze(1).clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('train_bdry/{}'.format(c), bdry[0, c, :, :].unsqueeze(0), global_step)

            # grid_image = make_grid(dt[:3, c, :, :].unsqueeze(1).clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('train_dt/{}'.format(c), dt[0, c, :, :].unsqueeze(0), global_step)


def val_epoch(model, data_loader, epoch, args, summary, device):
    model.eval()
    iterator = tqdm(data_loader)

    mIoU_ac, mAP_ac, mF1_ac = [], [], []
    mMask_ac = []
    for i, (image, mask, g_map) in enumerate(iterator):
        image = image.to(device)
        mask = mask.to(device)
        g_map = g_map.to(device)

        pred_mask_ac, disp, bdry, dt = model(image, args.iter)
        pred_mask_ac = F.interpolate(pred_mask_ac, size=mask.shape[1:], mode='bilinear')

        loss_masks_ac = F.mse_loss(pred_mask_ac.squeeze(1), mask)

        # Metrices
        iou_ac = get_iou(pred_mask_ac.gt(0.5), mask.bool())
        ap_ac = get_ap_scores(pred_mask_ac.gt(0.5), mask)
        f1_ac = get_f1_scores(pred_mask_ac.gt(0.5), mask)
        mIoU_ac += iou_ac
        mAP_ac += ap_ac
        mF1_ac += f1_ac
        mMask_ac += [loss_masks_ac.item()]

        iterator.set_description(
            '(val   | {}) Epoch [{epoch}/{epochs}] :: Loss AC {loss_ac:.4f}'.format(
                args.checkname + '_' + args.exp,
                epoch=epoch + 1,
                epochs=args.epochs,
                loss_ac=loss_masks_ac.item()))

        global_step = (epoch // args.eval_rate) * len(data_loader) + i
        # summary.add_scalar('val/loss_ac', loss_masks_ac.item(), global_step)
        # summary.add_scalar('val/iou_ac', np.mean(iou_ac), global_step)
        # summary.add_scalar('val/ap_ac', np.mean(ap_ac), global_step)

        ind = np.argwhere(np.array(iou_ac) < 0.5).flatten().tolist()
        summary.visualize_image('val',
                                image,
                                mask.unsqueeze(1),
                                pred_mask_ac,
                                pred_mask_ac,
                                global_step)
        if ind:
            summary.visualize_image('val_BAD',
                                    image[ind],
                                    mask.unsqueeze(1)[ind],
                                    pred_mask_ac[ind],
                                    pred_mask_ac[ind],
                                    global_step)
            grid_image = make_grid(disp[ind][:3, :1, :, :].clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('val_BAD/Disp_x', grid_image, global_step)
            grid_image = make_grid(disp[ind][:3, 1:, :, :].clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('val_BAD/Disp_y', grid_image, global_step)

            grid_image = make_grid(g_map[ind][:3, :1, :, :].clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('val_BAD/Gmap_x', grid_image, global_step)

            grid_image = make_grid(g_map[ind][:3, 1:, :, :].clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('val_BAD/Gmap_y', grid_image, global_step)

        grid_image = make_grid(disp[:3, :1, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('val/Disp_x', grid_image, global_step)
        grid_image = make_grid(disp[:3, 1:, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('val/Disp_y', grid_image, global_step)

        grid_image = make_grid(g_map[:3, :1, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('val/Gmap_x', grid_image, global_step)

        grid_image = make_grid(g_map[:3, 1:, :, :].clone().cpu().data, 3, normalize=True)
        summary.writer.add_image('val/Gmap_y', grid_image, global_step)

        for c in range(bdry.shape[1]):
            # grid_image = make_grid(bdry[:3, c, :, :].unsqueeze(1).clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('val_bdry/{}'.format(c), bdry[0, c, :, :].unsqueeze(0), global_step)

            # grid_image = make_grid(dt[:3, c, :, :].unsqueeze(1).clone().cpu().data, 3, normalize=True)
            summary.writer.add_image('val_dt/{}'.format(c), dt[0, c, :, :].unsqueeze(0), global_step)

    return np.mean(mIoU_ac), np.mean(mAP_ac), np.mean(mF1_ac), np.mean(mMask_ac)


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

    # Architecture
    parser.add_argument('--arch', type=str,
                        default='unet',
                        choices=['unet', 'resnet'],
                        help='Network architecture. unet or resnet')
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
    parser.add_argument('--stages',
                        type=int,
                        nargs='+',
                        default=[0, 1, 2, 3],
                        help='ResNet skip connections')
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

    # Cityscapes Data
    parser.add_argument('--inst-path', type=str,
                        default='/path/to/cityscapes_instances',
                        help='Path to Cityscapes instances directory')
    parser.add_argument('--ann-type', type=str,
                        default='full',
                        choices=['comp', 'full'],
                        help='Type of annotation, full instance or only components')
    parser.add_argument('--class-name', type=str,
                        default='rider',
                        help='Class for Cityscapes dataset')
    parser.add_argument('--loops', type=int,
                        default=10,
                        help='Data repetition in Cityscapes dataset')

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

    args.checkname = args.class_name + '_' + args.checkname + '_' + args.arch

    torch.multiprocessing.set_start_method('spawn')

    # Define Saver
    saver = Saver(args)
    saver.save_experiment_config()

    # Define Tensorboard Summary
    summary = TensorboardSummary(saver.experiment_dir)
    args.exp = saver.experiment_dir.split('_')[-1]

    if args.train_dataset == 'cityscapes':
        # Data
        train_trans = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.2, 2)),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(22, scale=(0.75, 1.25)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            # transforms.NormalizeInstance()
        ])
        val_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size), do_mask=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            # transforms.NormalizeInstance()
        ])

        if args.ann_type == 'comp':
            train_ds = CityscapesInstances_comp(args.inst_path,
                                                args.ann_train,
                                                args.class_name,
                                                transformations=train_trans,
                                                loops=args.loops)
        elif args.ann_type == 'full':
            train_ds = CityscapesInstances(args.inst_path,
                                           args.ann_train,
                                           args.class_name,
                                           transformations=train_trans,
                                           loops=args.loops)
        else:
            raise Exception('problem with annotation type')

        val_ds = CityscapesInstances(args.inst_path,
                                     args.ann_val,
                                     args.class_name,
                                     transformations=val_trans,
                                     loops=1)
    elif args.train_dataset == "buildings":
        # Data
        MEAN = np.array([0.47341759 * 255, 0.28791303 * 255, 0.2850705 * 255])
        STD = np.array([0.22645572 * 255, 0.15276193 * 255, 0.140702 * 255])
        train_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomAffineFromSet(degrees=[0, 15, 60, 90, 135, 180, 225, 270], scale=(0.75, 1.25)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            # transforms.NormalizeInstance(),
            transforms.Normalize(MEAN, STD),
        ])
        val_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
            # transforms.NormalizeInstance()
            transforms.Normalize(MEAN, STD),
        ])

        train_ds = BuildingsDataset(args.data_path,
                                    args.ann_train,
                                    transformations=train_trans)
        val_ds = BuildingsDataset(args.data_path,
                                  args.ann_val,
                                  transformations=val_trans)
    else:
        # Data
        MEAN = np.array([0.47341759 * 255, 0.28791303 * 255, 0.2850705 * 255])
        STD = np.array([0.22645572 * 255, 0.15276193 * 255, 0.140702 * 255])
        train_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomAffineFromSet(degrees=[0, 15, 60, 90, 135, 180, 225, 270], scale=(0.75, 1.25)),
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

    model = networks.CircleNet(args).to(device)

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
            train_epoch(model, optimizer, train_dl, epoch, args, summary, device)

        if epoch % args.eval_rate == 0:
            mIoU_ac, mAP_ac, mF1_ac, mMask_ac = val_epoch(model, val_dl, epoch, args, summary, device)

            global_step = epoch // args.eval_rate
            summary.add_scalar('val/mIoU_ac', mIoU_ac, global_step)
            summary.add_scalar('val/mAP_ac', mAP_ac, global_step)
            summary.add_scalar('val/mF1_ac', mF1_ac, global_step)
            summary.add_scalar('val/mMask_ac', mMask_ac, global_step)

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
