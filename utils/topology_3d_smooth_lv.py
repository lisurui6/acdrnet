import torch
import math
import open3d as o3d
import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
from rasterizor.voxelize import Voxelize


def batch_sample_lv_ball_points(
    c0x, c0y, c0z, r1, n, voxel_width, voxel_height, voxel_depth
):
    """

    Args:
        c0x: (B,)
        c0y: (B,)
        c0z: (B,)
        r1: (B,)

    Returns:

    """
    c0x = (c0x.unsqueeze(1).unsqueeze(1) / 2 + 0.5) * voxel_width  # (B, 1, 1)
    c0y = (c0y.unsqueeze(1).unsqueeze(1) / 2 + 0.5) * voxel_height  # (B, 1, 1)
    c0z = (c0z.unsqueeze(1).unsqueeze(1) / 2 + 0.5) * voxel_depth  # (B, 1, 1)
    r1 = r1.unsqueeze(1).unsqueeze(1) * voxel_width  # (B, 1)

    batch_size = c0x.shape[0]
    theta = torch.arange(1, n).cuda()
    theta = math.pi * theta / n

    phi = torch.arange(2*n).cuda()
    phi = 2 * math.pi * phi / (2*n)
    theta_grid, phi_grid = torch.meshgrid(theta, phi)  # (2n, n)
    theta_grid = theta_grid.unsqueeze(0).repeat(batch_size, 1, 1)
    phi_grid = phi_grid.unsqueeze(0).repeat(batch_size, 1, 1)

    x = torch.mul(r1, torch.mul(torch.cos(phi_grid), torch.sin(theta_grid))) + c0x  # (B, 2n, n)
    y = torch.mul(r1, torch.mul(torch.sin(phi_grid), torch.sin(theta_grid))) + c0y
    z = torch.mul(r1, torch.cos(theta_grid)) + c0z

    x = x.view(batch_size, -1)  # (B, 2n * n)
    y = y.view(batch_size, -1)
    z = z.view(batch_size, -1)

    points = torch.stack([x, y, z], dim=2)
    top_point = torch.cat([c0x, c0y, c0z + r1], dim=2)
    end_point = torch.cat([c0x, c0y, c0z - r1], dim=2)
    points = torch.cat([points, top_point, end_point], dim=1)

    center = torch.cat([c0x, c0y, c0z], dim=2)
    # print(center, "center")
    # print(r1, "r1")
    # points = torch.cat([points, center], dim=1)
    return points


def batch_sample_lv_myo_points(
        c0x, c0y, c0z, r1, r0_r1_ratio,
        dz, dx_dz, dy_dz, dr1, dr0,
        num_points_per_slice, num_lv_slices,
        voxel_width, voxel_height, voxel_depth, batch_size
):
    """

    Args:
        c0x: (B,)
        c0y: (B,)
        c0z: (B,)
        r1: (B,)
        r0_r1_ratio: (B,)
        dz: (B, n_lv - 1), (0, 1) * (voxel_depth - c0z) / (num_extra_lv_myo_slices + num_lv_slices)
        dx_dz: (B, n_lv - 1), dx/dz, (-1, 1) * d_max
        dy_dz: (B, n_lv - 1), dx/dz, (-1, 1) * d_max
        dr0: (B, n_lv - 1), dr0/dr0_max, (-1, 1) * dr0_max
        dr1: (B, n_lv + n_myo - 1), dr1/dr1_max, (-1, 1) * dr1_max
        num_points_per_slice:
        num_lv_slices:
        num_extra_ly_myo_slices:
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

    dz = torch.mul(dz, (voxel_depth - c0z) / (num_lv_slices))  # (B, n_lv + n_myo - 1)
    c0z = c0z + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dz, dim=1)], dim=1)  # (B, n_lv + n_myo)

    dx = torch.mul(dx_dz, dz) * d_max  # (B, n_lv + n_myo - 1)
    c0x = c0x + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dx, dim=1)], dim=1)  # (B, n_lv + n_myo)

    dy = torch.mul(dy_dz, dz) * d_max  # (B, n_lv + n_myo - 1)
    c0y = c0y + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dy, dim=1)], dim=1)  # (B, n_lv + n_myo)

    r1 = r1.unsqueeze(1) * voxel_width  # (B, 1)
    dr1_max = r1 / (num_lv_slices - 1)  # (B, 1)
    dr1 = (dr1 -1) / 2
    dr1 = torch.mul(dr1, dr1_max)  # (B, n_lv + n_myo - 1)
    r1 = r1 + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dr1, dim=1)], dim=1)  # (B, n_lv + n_myo)

    r0 = torch.mul(r0_r1_ratio.unsqueeze(1), r1[:, :num_lv_slices])  # (B, n_lv)
    dr0 = (dr0 - 1) / 2
    dr0_max = r0[:, 0].unsqueeze(1) / (num_lv_slices - 1)  # (B, 1)
    dr0 = torch.mul(dr0, dr0_max)  # (B, n_lv - 1)
    r0 = r0 + torch.cat([torch.zeros(batch_size, 1).cuda(), torch.cumsum(dr0, dim=1)], dim=1)  # (B, n_lv)

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
        o3d.visualization.draw_geometries([lv_pcd, lv_tetra_mesh], mesh_show_back_face=False)

    lv_points[..., 0] = (lv_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_points[..., 1] = (lv_points[..., 1] - voxel_height / 2) / voxel_height * 2
    lv_points[..., 2] = (lv_points[..., 2] - voxel_depth / 2) / voxel_depth * 2

    return lv_points


def o3d_volumetric_mesh(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    convex_mesh, __ = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    return pcd, convex_mesh


def visualise_o3d_mesh(
        mesh, pcd, is_triangle: bool = False, show_voxel: bool = False, voxel_size=0.01, show_pcd_normal: bool = True,
        use_mayavi: bool = False, fname: str = "mesh",
):
    o3d.visualization.draw_geometries([pcd], point_show_normal=show_pcd_normal)
    points = np.asarray(mesh.vertices)
    o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=False)

    if is_triangle:
        if use_mayavi:
            mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], np.asarray(mesh.triangles))
            mlab.savefig(fname + ".png")
            mlab.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_trisurf(
                points[:, 0], points[:, 1], -points[:, 2], triangles=np.asarray(mesh.triangles), cmap=plt.cm.Spectral
            )
            print("saving {}.png".format(fname))
            plt.savefig(fname + ".png")
    if show_voxel:
        if is_triangle:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
        else:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
        o3d.visualization.draw_geometries([voxel_grid])


def sample_3d_points(
    par1, par2, num_lv_slices, voxel_width, voxel_height, voxel_depth,
    num_points, batch_size, epoch, lv_tetras=None, lv_myo_tetras=None, rv_tetras=None, vis=False
):
    """
    par1:
        c0_x, c0_y, c0_z,
    par2:
        r1
    dz = (0, 1) * (1 - c0_z) * depth / (num_extra_lv_myo_slices + num_lv_slices)
    dx = dxy/dz (-1, 1) * dz
    """
    lv_points = batch_sample_lv_ball_points(
        c0x=par1[:, 0],
        c0y=par1[:, 1],
        c0z=par1[:, 2],
        r1=par2[:, 0],
        n=num_points,
        voxel_width=voxel_width,
        voxel_height=voxel_height,
        voxel_depth=voxel_depth,
    )

    if False:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lv_points[0].detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[255, 0, 0]]), (lv_points.shape[1], 1))
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    if lv_tetras is None or vis:
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        lv_tetras = lv_tetra_mesh.tetras
        visualise_o3d_mesh(
            mesh=lv_tetra_mesh,
            pcd=lv_pcd,
            is_triangle=False,
            show_pcd_normal=False,
            show_voxel=False,
            voxel_size=0.02,
            fname="lv_ball_tetra"
        )
    print(lv_points)
    lv_points[..., 0] = (lv_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_points[..., 1] = (lv_points[..., 1] - voxel_height / 2) / voxel_height * 2
    lv_points[..., 2] = (lv_points[..., 2] - voxel_depth / 2) / voxel_depth * 2
    print(lv_points)
    batch_lv_tetras = torch.Tensor(lv_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)

    return lv_points, batch_lv_tetras, lv_tetras

