import math
import torch
import numpy as np
import open3d as o3d


def o3d_volumetric_mesh(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    convex_mesh, __ = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    return pcd, convex_mesh


def batch_sample_lv_myo_points(
    c0x, c0y, c0z, c0z_end, r0, dx, dy, dr, num_points_per_slice, num_lv_slices,
    voxel_width, voxel_height, voxel_depth, batch_size
):
    """

    Args:
        c0x: (B,)
        c0y: (B,)
        c0z: (B,)
        c0z_end: (B,)
        r0: (B,)
        dx: (B, n_lv - 1)
        dy: (B, n_lv - 1)
        dr: (B, n_lv - 1)
        num_points_per_slice:
        num_lv_slices:
        voxel_width:
        voxel_height:
        voxel_depth:
        batch_size:

    Returns:

    """
    d_max = 0.5
    batch_size = c0x.shape[0]
    c0x = (c0x.unsqueeze(1) / 2 + 0.5) * voxel_width  # (B, 1)
    c0y = (c0y.unsqueeze(1) / 2 + 0.5) * voxel_height  # (B, 1)
    c0z = (c0z.unsqueeze(1) / 2 + 0.5) * voxel_depth  # (B, 1)
    c0z_end = (c0z_end.unsqueeze(1) / 2 + 0.5) * voxel_depth  # (B, 1)

    dz = (c0z_end - c0z) / num_lv_slices  # (B, 1)
    dz = dz.repeat(1, num_lv_slices - 1)  # (B, n_lv - 1)
    c0z = c0z + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dz, dim=1)], dim=1)  # (B, n_lv)

    dx = dx * d_max  # (B, n_lv - 1)
    c0x = c0x + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dx, dim=1)], dim=1)  # (B, n_lv)

    dy = dy * d_max  # (B, n_lv - 1)
    c0y = c0y + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dy, dim=1)], dim=1)  # (B, n_lv)

    r0 = r0.unsqueeze(1) * voxel_width  # (B, 1)
    dr0_max = r0 / (num_lv_slices - 1)  # (B, 1)
    dr0 = torch.mul(dr, dr0_max)  # (B, n_lv - 1)
    r0 = r0 - torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dr0, dim=1)], dim=1)  # (B, n_lv)

    c0_phase = torch.arange(num_points_per_slice).repeat(batch_size, num_lv_slices, 1).cuda()
    c0_phase = 2 * math.pi * c0_phase / num_points_per_slice  # (B, n_lv, n_points)
    c0_angle = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().repeat(batch_size, num_lv_slices, num_points_per_slice).cuda(),
            imag=c0_phase,
        )
    )  # (B, n_lv, n_points)
    z_c0 = torch.complex(real=c0x[:, :num_lv_slices], imag=c0y[:, :num_lv_slices]).unsqueeze(2)  # (B, n_lv, 1)
    lv_xy = z_c0.repeat(1, 1, num_points_per_slice) + torch.mul(r0.unsqueeze(2).repeat(1, 1, num_points_per_slice), c0_angle)  # (B, n_lv, n_points)
    lv_xy = torch.cat([lv_xy, z_c0], dim=2)
    lv_xy = torch.view_as_real(lv_xy)  # (B, n_lv, n_points, 2)

    lv_points = torch.cat([lv_xy, c0z[:, :num_lv_slices].unsqueeze(2).repeat(1, 1, num_points_per_slice + 1).unsqueeze(3)], dim=3)

    lv_points = lv_points.view(lv_points.shape[0], -1, lv_points.shape[3])

    plot = False
    if plot:
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        o3d.visualization.draw_geometries([lv_pcd], point_show_normal=True)
        points = np.asarray(lv_tetra_mesh.vertices)
        o3d.visualization.draw_geometries([lv_pcd, lv_tetra_mesh], mesh_show_back_face=False)


    lv_points[..., 0] = (lv_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_points[..., 1] = (lv_points[..., 1] - voxel_height / 2) / voxel_height * 2
    lv_points[..., 2] = (lv_points[..., 2] - voxel_depth / 2) / voxel_depth * 2

    return lv_points


def sample_lv_points(
    par1, par2, par3, num_lv_slices, voxel_width, voxel_height, voxel_depth,
    num_points, batch_size, epoch, lv_tetras=None,
):
    """
    par1:
        c0_x, c0_y, c0_z, c0_z_end
    par2:
        r0
    par3:
        dx, dy, dr
    dz = (0, 1) * torch.mul(dz, (c0z_end - c0z) * depth / (num_lv_slices))
    dx = [-1, 1] * d_max
    """

    d_slices = num_lv_slices - 1
    lv_points = batch_sample_lv_myo_points(
        c0x=par1[:, 0],
        c0y=par1[:, 1],
        c0z=par1[:, 2],
        c0z_end=par1[:, 3],
        r0=par2[:, 0],
        dx=par3[:, :d_slices],
        dy=par3[:, d_slices:2*d_slices],
        dr=par3[:, 2*d_slices:3*d_slices],
        voxel_width=voxel_width,
        voxel_height=voxel_height,
        voxel_depth=voxel_depth,
        num_points_per_slice=num_points,
        num_lv_slices=num_lv_slices,
        batch_size=batch_size,
    )

    if epoch < -1:
        pcd = o3d.geometry.PointCloud()
        points = lv_points
        pcd.points = o3d.utility.Vector3dVector(points[0].detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(
            np.concatenate(
                [
                    np.tile(np.array([[255, 0, 0]]), (lv_points.shape[1], 1)),
                ],
                axis=0,
            )
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    if lv_tetras is None:
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        lv_tetras = lv_tetra_mesh.tetras

    batch_lv_tetras = torch.Tensor(lv_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)

    return lv_points, batch_lv_tetras, lv_tetras