import torch
import math
import open3d as o3d
import numpy as np
from mayavi import mlab
from matplotlib import pyplot as plt
from rasterizor.voxelize import Voxelize


def sample_rv_outer_arc(theta_c2, theta2, num_points, batch_size, z_c0, z_c2, r1):
    theta_p0 = theta_c2 - theta2  # theta_p0 = (-pi, 2pi)
    theta_p1 = theta_c2 + theta2  # theta_p1 = (0, 3pi)

    # theta_p0, theta_p1 = (0, 2pi)
    theta_p0 = torch.where(
        theta_p0 < 0,
        theta_p0 + math.pi * 2,
        theta_p0,
    )

    theta_p1 = torch.where(
        theta_p1 > math.pi * 2,
        theta_p1 - math.pi * 2,
        theta_p1,
    )

    theta_p1 = torch.where(
        theta_p1 < theta_p0,
        theta_p1 + math.pi * 2,
        theta_p1
    )
    n_arc_points = num_points // 2

    theta_p0 = theta_p0.repeat(n_arc_points, 1).transpose(1, 0)
    theta_p1 = theta_p1.repeat(n_arc_points, 1).transpose(1, 0)

    arc_count = torch.arange(n_arc_points).repeat(batch_size, 1).cuda()
    arc_phase = theta_p0 + torch.mul(theta_p1 - theta_p0, arc_count) / (n_arc_points - 1)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().repeat(batch_size, n_arc_points).cuda(), imag=arc_phase)
    )
    arc = z_c0.repeat(n_arc_points, 1).transpose(1, 0) + r1.repeat(n_arc_points, 1).transpose(1, 0) * arc_angle
    arc_1 = torch.flip(arc, dims=[1])  # p1 to p0 arc

    r2 = (torch.view_as_real(z_c2) - torch.view_as_real(arc_1[:, -1])).norm(dim=1)
    theta_c2_p0 = torch.log(arc_1[:, -1] - z_c2).imag  # theta_c2_p0 = (-pi, pi)
    theta_c2_p1 = torch.log(arc_1[:, 0] - z_c2).imag  # theta_c2_p1 = (-pi, pi)

    theta_c2_p0 = torch.where(
        theta_c2_p0 < 0,
        theta_c2_p0 + math.pi * 2,
        theta_c2_p0,
    )

    theta_c2_p1 = torch.where(
        theta_c2_p1 < 0,
        theta_c2_p1 + math.pi * 2,
        theta_c2_p1,
    )

    theta_c2_p1 = torch.where(
        theta_c2_p0 > theta_c2_p1,
        theta_c2_p1 + math.pi * 2,
        theta_c2_p1,
    )
    theta_c2_p0 = theta_c2_p0.repeat(n_arc_points, 1).transpose(1, 0)
    theta_c2_p1 = theta_c2_p1.repeat(n_arc_points, 1).transpose(1, 0)

    arc_phase = theta_c2_p1 + torch.mul(theta_c2_p0 - theta_c2_p1, arc_count) / (n_arc_points - 1)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().repeat(batch_size, n_arc_points).cuda(), imag=arc_phase))
    arc_2 = z_c2.repeat(n_arc_points, 1).transpose(1, 0) + r2.repeat(n_arc_points, 1).transpose(1, 0) * arc_angle
    return arc_1, arc_2


def batch_sample_lv_myo_points(
        c0x, c0y, c0z, r1, r0_r1_ratio, dz,
        num_points_per_slice, num_lv_slices, num_extra_ly_myo_slices,
        voxel_width, voxel_height, voxel_depth, batch_size
):
    """

    Args:
        c0x: (B, n_lv + n_myo)
        c0y: (B, n_lv + n_myo)
        c0z: (B, 1)
        r1: (B, n_lv + n_myo)
        r0_r1_ratio: (B, n_lv)
        dz: (B, n_lv + n_myo - 1), (0, 1) * (voxel_depth - c0z) / (num_extra_lv_myo_slices + num_lv_slices)
        num_points_per_slice:
        num_lv_slices:
        num_extra_ly_myo_slices:
        voxel_width:
        voxel_height:
        voxel_depth:
        batch_size:

    Returns:

    """
    c0x = (c0x / 2 + 0.5) * voxel_width  # (B, n_lv + n_myo)
    c0y = (c0y / 2 + 0.5) * voxel_height  # (B, n_lv + n_myo)
    c0z = (c0z.unsqueeze(1) / 2 + 0.5) * voxel_depth  # (B, 1)
    dz = torch.mul(dz, (voxel_depth - c0z) / (num_extra_ly_myo_slices + num_lv_slices))  # (B, n_lv + n_myo - 1)
    c0z = c0z + torch.cat([torch.zeros(dz.shape[0], 1).cuda(), torch.cumsum(dz, dim=1)], dim=1)  # (B, n_lv + n_myo)

    r1 = r1 * voxel_width  # (B, n_lv + n_myo)
    r0 = torch.mul(r0_r1_ratio, r1[:, :num_lv_slices])  # (B, n_lv)

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

    # myo
    z_c1 = torch.complex(real=c0x, imag=c0y).unsqueeze(2)  # (B, n_lv + n_myo, 1)
    c1_phase = torch.arange(num_points_per_slice).repeat(batch_size, num_lv_slices + num_extra_ly_myo_slices, 1).cuda()
    c1_phase = 2 * math.pi * c1_phase / num_points_per_slice  # (B, n_lv + n_myo, n_points)
    c1_angle = torch.exp(
        torch.complex(
            real=torch.tensor(0).float().repeat(batch_size, num_lv_slices + num_extra_ly_myo_slices, num_points_per_slice).cuda(),
            imag=c1_phase,
        )
    )  # (B, n_lv + n_myo, n_points)
    myo_xy = z_c1.repeat(1, 1, num_points_per_slice) + torch.mul(r1.unsqueeze(2).repeat(1, 1, num_points_per_slice), c1_angle)  # (B, n_lv + n_myo, n_points)
    myo_xy = torch.cat([myo_xy, z_c1], dim=2)
    myo_xy = torch.view_as_real(myo_xy)  # (B, n_lv, n_points, 2)
    lv_myo_points = torch.cat([myo_xy, c0z.unsqueeze(2).repeat(1, 1, num_points_per_slice + 1).unsqueeze(3)], dim=3)

    lv_points = lv_points.view(lv_points.shape[0], -1, lv_points.shape[3])
    lv_myo_points = lv_myo_points.view(lv_myo_points.shape[0], -1, lv_myo_points.shape[3])


    plot = False
    if plot:
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        o3d.visualization.draw_geometries([lv_pcd], point_show_normal=True)
        points = np.asarray(lv_tetra_mesh.vertices)
        o3d.visualization.draw_geometries([lv_pcd, lv_tetra_mesh], mesh_show_back_face=False)

        lv_myo_pcd, lv_myo_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_myo_points[0].detach().cpu().numpy(),
        )
        o3d.visualization.draw_geometries([lv_myo_pcd], point_show_normal=True)
        points = np.asarray(lv_myo_tetra_mesh.vertices)
        o3d.visualization.draw_geometries([lv_myo_pcd, lv_myo_tetra_mesh], mesh_show_back_face=False)

    # points = np.asarray(mesh.vertices)
    # o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=False)

    lv_points[..., 0] = (lv_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_points[..., 1] = (lv_points[..., 1] - voxel_height / 2) / voxel_height * 2
    lv_points[..., 2] = (lv_points[..., 2] - voxel_depth / 2) / voxel_depth * 2

    lv_myo_points[..., 0] = (lv_myo_points[..., 0] - voxel_width / 2) / voxel_width * 2
    lv_myo_points[..., 1] = (lv_myo_points[..., 1] - voxel_height / 2) / voxel_height * 2
    lv_myo_points[..., 2] = (lv_myo_points[..., 2] - voxel_depth / 2) / voxel_depth * 2

    return lv_points, lv_myo_points


def batch_sample_rv_points(
        c0x, c0y, c0z, dz, r1, theta_c2, theta2_ratio, d_c2_c0_ratio, num_points_per_slice, num_slices,
        voxel_width, voxel_height, voxel_depth, batch_size
):
    """

    Args:
        c0x: (B, n_lv)
        c0y: (B, n_lv)
        d0z: (B,)
        dz: (B, n_lv - 1)
        r1: (B, n_lv)
        theta_c2: (B, n_lv)
        theta2_ratio: (B, n_lv)
        d_c2_c0_ratio: (B, n_lv)
        num_points_per_slice:
        num_slices: n_lv
        voxel_width:
        voxel_height:
        voxel_depth:
        batch_size:

    Returns:

    """
    c0x = (c0x / 2 + 0.5) * voxel_width
    c0y = (c0y / 2 + 0.5) * voxel_height

    c0z = (c0z.unsqueeze(1) / 2 + 0.5) * voxel_depth  # (B, 1)
    dz = torch.mul(dz, (voxel_depth - c0z) / num_slices)  # (B, n_lv - 1)
    c0z = c0z + torch.cat([torch.zeros(dz.shape[0], 1).cuda(), torch.cumsum(dz, dim=1)], dim=1)  # (B, n_lv)

    r1 = r1 * voxel_width

    theta2_max = torch.tensor(math.pi * 3 / 4).float().repeat(batch_size).cuda().unsqueeze(1)
    theta2_min = torch.tensor(math.pi * 1 / 6).float().repeat(batch_size).cuda().unsqueeze(1)
    theta2 = theta2_min + torch.mul(theta2_ratio, theta2_max - theta2_min)  # (B, n_lv)
    z_c0 = torch.complex(real=c0x, imag=c0y)  # (B, n_lv)
    dmin = r1 * 1 / 2  # (B, n_lv)
    dmax = r1 * 3 / 2  # (B, n_lv)
    d_c2_c0 = torch.mul(d_c2_c0_ratio, dmax - dmin) + dmin  # (B, n_lv)
    theta_c2 = theta_c2 * math.pi * 2  # (B, n_lv)

    z_c2 = z_c0 + d_c2_c0 * torch.exp(
        torch.complex(real=torch.tensor(0).float().cuda().repeat(batch_size, num_slices), imag=theta_c2.float())
    )  # (B, n_lv)

    theta_p0 = theta_c2 - theta2  # theta_p0 = (-pi, 2pi), (B, n_lv)
    theta_p1 = theta_c2 + theta2  # theta_p1 = (0, 3pi), (B, n_lv)

    # theta_p0, theta_p1 = (0, 2pi)
    theta_p0 = torch.where(
        theta_p0 < 0,
        theta_p0 + math.pi * 2,
        theta_p0,
    )  # (B, n_lv)

    theta_p1 = torch.where(
        theta_p1 > math.pi * 2,
        theta_p1 - math.pi * 2,
        theta_p1,
    )  # (B, n_lv)

    theta_p1 = torch.where(
        theta_p1 < theta_p0,
        theta_p1 + math.pi * 2,
        theta_p1
    )  # (B, n_lv)
    n_arc_points = num_points_per_slice // 2

    theta_p0 = theta_p0.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_lv, n_arc_points)
    theta_p1 = theta_p1.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_lv, n_arc_points)

    arc_count = torch.arange(n_arc_points).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_slices, 1).cuda()  # (B, n_lv, n_arc_points)
    arc_phase = theta_p0 + torch.mul(theta_p1 - theta_p0, arc_count) / (n_arc_points - 1)  # (B, n_lv, n_arc_points)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().repeat(batch_size, num_slices, n_arc_points).cuda(), imag=arc_phase)
    )  # (B, n_lv, n_arc_points)
    arc = z_c0.unsqueeze(2).repeat(1, 1, n_arc_points) + torch.mul(r1.unsqueeze(2).repeat(1, 1, n_arc_points), arc_angle)  # (B, n_lv, n_arc_points)
    arc_1 = torch.flip(arc, dims=[2])  # p1 to p0 arc

    r2 = (torch.view_as_real(z_c2) - torch.view_as_real(arc_1[..., -1])).norm(dim=2)  # (B, n_lv)
    theta_c2_p0 = torch.log(arc_1[..., -1] - z_c2).imag  # theta_c2_p0 = (-pi, pi), (B, n_lv)
    theta_c2_p1 = torch.log(arc_1[..., 0] - z_c2).imag  # theta_c2_p1 = (-pi, pi), (B, n_lv)

    theta_c2_p0 = torch.where(
        theta_c2_p0 < 0,
        theta_c2_p0 + math.pi * 2,
        theta_c2_p0,
    )

    theta_c2_p1 = torch.where(
        theta_c2_p1 < 0,
        theta_c2_p1 + math.pi * 2,
        theta_c2_p1,
    )

    theta_c2_p1 = torch.where(
        theta_c2_p0 > theta_c2_p1,
        theta_c2_p1 + math.pi * 2,
        theta_c2_p1,
    )
    theta_c2_p0 = theta_c2_p0.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_lv, n_arc_points)
    theta_c2_p1 = theta_c2_p1.unsqueeze(2).repeat(1, 1, n_arc_points)  # (B, n_lv, n_arc_points)

    arc_phase = theta_c2_p1 + torch.mul(theta_c2_p0 - theta_c2_p1, arc_count) / (n_arc_points - 1)  # (B, n_lv, n_arc_points)
    arc_angle = torch.exp(
        torch.complex(real=torch.tensor(0).float().repeat(batch_size, num_slices, n_arc_points).cuda(), imag=arc_phase)
    )  # (B, n_lv, n_arc_points)
    arc_2 = z_c2.unsqueeze(2).repeat(1, 1, n_arc_points) + torch.mul(r2.unsqueeze(2).repeat(1, 1, n_arc_points), arc_angle)  # (B, n_lv, n_arc_points)
    arc_1 = torch.view_as_real(arc_1)  # (B, n_lv, n_arc_points, 2)
    arc_1 = arc_1.view(arc_1.shape[0], -1, arc_1.shape[-1])  # (B, n_lv * n_arc_points, 2)
    arc_2 = torch.view_as_real(arc_2)  # (B, n_lv, n_arc_points, 2)
    arc_2 = arc_2.view(arc_2.shape[0], -1, arc_2.shape[-1])  # (B, n_lv * n_arc_points, 2)

    c0z = c0z.unsqueeze(2).repeat(1, 1, n_arc_points).unsqueeze(3)  # (B, n_lv, n_arc_points, 1)
    c0z = c0z.view(c0z.shape[0], -1, c0z.shape[-1])  # (B, n_lv * n_arc_points, 1)
    surface2_arc_in = torch.cat([arc_1, c0z], dim=2)
    surface2_arc_out = torch.cat([arc_2, c0z], dim=2)

    surface2_arc_in[..., 0] = (surface2_arc_in[..., 0] - voxel_width / 2) / voxel_width * 2
    surface2_arc_in[..., 1] = (surface2_arc_in[..., 1] - voxel_height / 2) / voxel_height * 2
    surface2_arc_in[..., 2] = (surface2_arc_in[..., 2] - voxel_depth / 2) / voxel_depth * 2

    surface2_arc_out[..., 0] = (surface2_arc_out[..., 0] - voxel_width / 2) / voxel_width * 2
    surface2_arc_out[..., 1] = (surface2_arc_out[..., 1] - voxel_height / 2) / voxel_height * 2
    surface2_arc_out[..., 2] = (surface2_arc_out[..., 2] - voxel_depth / 2) / voxel_depth * 2
    rv_points = torch.cat([surface2_arc_in, surface2_arc_out], dim=1)

    return rv_points


def o3d_volumetric_mesh(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    convex_mesh, __ = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    return pcd, convex_mesh


def rv_volumentric_tetra(num_total_points, num_points_per_layer):
    """RV points are concated as [arc_in, arc_out]"""
    num_layers = int(num_total_points / num_points_per_layer)
    arc_out_offset = int(num_total_points / 2)
    num_points_per_layer = int(num_points_per_layer / 2)
    tetras = np.concatenate(
        [
            np.array([
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer],
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer + 1],
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer + arc_out_offset],
                [i + arc_out_offset, i + 1 + arc_out_offset, i, i + num_points_per_layer + arc_out_offset + 1],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer + 1],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer + arc_out_offset],
                [i + arc_out_offset + 1, i, i + 1, i + num_points_per_layer + arc_out_offset + 1],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i + 1],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i + arc_out_offset],
                [i + arc_out_offset + num_points_per_layer, i + 1 + arc_out_offset + num_points_per_layer, i + num_points_per_layer, i + arc_out_offset + 1],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i + 1],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i + arc_out_offset],
                [i + arc_out_offset + 1 + num_points_per_layer, i + num_points_per_layer, i + 1 + num_points_per_layer, i + 1 + arc_out_offset],
            ])
            for i in range(num_points_per_layer - 1)
        ], axis=0
    )
    print(tetras.shape)
    tetras = np.concatenate([tetras + i * num_points_per_layer for i in range(num_layers - 1)])
    print(tetras.shape)
    return tetras


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
    par1, par2, num_lv_slices, num_extra_lv_myo_slices, voxel_width, voxel_height, voxel_depth,
    num_points, batch_size, epoch, lv_tetras=None, lv_myo_tetras=None, rv_tetras=None,
):
    """
    par1: c0_x * (lv + myo), c0_y * (lv + myo) for each slice and c0_z for the first slice
    par2: [factor 0 (r0/r1)] * n_lv, [theta2/pi] * (lv + myo)), [d_c2_c0] * (lv + myo), [theta_c2] * (lv + myo) + [r1] * (lv + myo) + dz [lv, extra]
    dz = (0, 1) * (1 - c0_z) * depth / (num_extra_lv_myo_slices + num_lv_slices)
    """
    # width = 128
    # height = 128
    # depth = 64
    # batch_size = 1
    # par1 = torch.zeros(batch_size, 2).cuda().float()
    # par2 = torch.zeros(batch_size, 5).cuda().float()
    # par2[:, 0] = 32 / 128
    # par2[:, 1] = 2 / 3
    # par2[:, 2] = 1 / 2
    # par2[:, 3] = 33 / 128
    # par2[:, 4] = 1 / 4

    lv_points, lv_myo_points = batch_sample_lv_myo_points(
        c0x=par1[:, :(num_lv_slices + num_extra_lv_myo_slices)],
        c0y=par1[:, (num_lv_slices + num_extra_lv_myo_slices):-1],
        c0z=par1[:, -1],
        dz=par2[:, -(num_lv_slices + num_extra_lv_myo_slices - 1):],
        r1=par2[:, (num_lv_slices * 4 + num_extra_lv_myo_slices * 3): (num_lv_slices * 4 + num_extra_lv_myo_slices * 3 + num_lv_slices + num_extra_lv_myo_slices)],
        r0_r1_ratio=par2[:, :num_lv_slices],
        voxel_width=voxel_width,
        voxel_height=voxel_height,
        voxel_depth=voxel_depth,
        num_points_per_slice=num_points,
        num_lv_slices=num_lv_slices,
        num_extra_ly_myo_slices=num_extra_lv_myo_slices,
        batch_size=batch_size,
    )
    print("LV")
    print(lv_points.shape, lv_myo_points.shape)
    rv_points = batch_sample_rv_points(
        c0x=par1[:, :(num_lv_slices + num_extra_lv_myo_slices)],
        c0y=par1[:, (num_lv_slices + num_extra_lv_myo_slices):-1],
        c0z=par1[:, -1],
        dz=par2[:, -(num_lv_slices + num_extra_lv_myo_slices - 1):],
        r1=par2[:, (num_lv_slices * 4 + num_extra_lv_myo_slices * 3): (num_lv_slices * 4 + num_extra_lv_myo_slices * 3 + num_lv_slices + num_extra_lv_myo_slices)],
        theta_c2=par2[:, (3 * num_lv_slices + 2 * num_extra_lv_myo_slices):(4 * num_lv_slices + 3 * num_extra_lv_myo_slices)],
        theta2_ratio=par2[:, num_lv_slices:(2 * num_lv_slices + num_extra_lv_myo_slices)],
        d_c2_c0_ratio=par2[:, (2 * num_lv_slices + num_extra_lv_myo_slices):(3 * num_lv_slices + 2 * num_extra_lv_myo_slices)],
        voxel_width=voxel_width,
        voxel_height=voxel_height,
        voxel_depth=voxel_depth,
        num_points_per_slice=num_points,
        num_slices=num_lv_slices + num_extra_lv_myo_slices,
        batch_size=batch_size,
    )
    if epoch > 30:
        pcd = o3d.geometry.PointCloud()
        points = torch.cat([lv_points, lv_myo_points, rv_points], dim=1)
        pcd.points = o3d.utility.Vector3dVector(points[0].detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(
            np.concatenate(
                [
                    np.tile(np.array([[255, 0, 0]]), (lv_points.shape[1], 1)),
                    np.tile(np.array([[0, 255, 0]]), (lv_myo_points.shape[1], 1)),
                    np.tile(np.array([[0, 0, 255]]), (rv_points.shape[1], 1)),
                ],
                axis=0,
            )
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    if lv_myo_tetras is None:

        lv_myo_pcd, lv_myo_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_myo_points[0].detach().cpu().numpy(),
        )
        print(np.asarray(lv_myo_tetra_mesh.tetras).shape, "tetras lv")
        lv_myo_tetras = lv_myo_tetra_mesh.tetras

    if lv_tetras is None:
        lv_pcd, lv_tetra_mesh = o3d_volumetric_mesh(
            vertices=lv_points[0].detach().cpu().numpy(),
        )
        lv_tetras = lv_tetra_mesh.tetras

    batch_lv_tetras = torch.Tensor(lv_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)
    batch_lv_myo_tetras = torch.Tensor(lv_myo_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)

    # visualise_o3d_mesh(
    #     mesh=lv_myo_tetra_mesh,
    #     pcd=lv_myo_pcd,
    #     is_triangle=False,
    #     show_pcd_normal=False,
    #     show_voxel=False,
    #     voxel_size=0.02,
    #     fname="lv_myo_tetra"
    # )
    print("doing RV...")
    # rv_points_arc_in_pcd = o3d.geometry.PointCloud()
    # rv_points_arc_in_pcd.points = o3d.utility.Vector3dVector(rv_points_arc_in[0].detach().cpu().numpy())
    # o3d.visualization.draw_geometries([rv_points_arc_in_pcd], point_show_normal=False)
    #
    # rv_points_arc_out_pcd = o3d.geometry.PointCloud()
    # rv_points_arc_out_pcd.points = o3d.utility.Vector3dVector(rv_points_arc_out[0].detach().cpu().numpy())
    # o3d.visualization.draw_geometries([rv_points_arc_out_pcd], point_show_normal=False)

    if rv_tetras is None:
        rv_tetras = rv_volumentric_tetra(rv_points.shape[1], num_points)
    batch_rv_tetras = torch.Tensor(rv_tetras)[None, None, ...].to(torch.device("cuda")).repeat(batch_size, 1, 1, 1).type(torch.int32)

    return lv_points, batch_lv_tetras, lv_myo_points, batch_lv_myo_tetras, rv_points, batch_rv_tetras, \
           lv_tetras, lv_myo_tetras, rv_tetras


# sparse_rv_pcd, convex_rv_mesh = o3d_volumetric_mesh(rv_points[0].detach().cpu().numpy())
# sparse_rv_pcd = o3d.geometry.PointCloud()
# sparse_rv_pcd.points = o3d.utility.Vector3dVector(rv_points[0].detach().cpu().numpy())
# mesh = o3d.geometry.TetraMesh(o3d.utility.Vector3dVector(rv_points[0].detach().cpu().numpy()), o3d.utility.Vector4iVector(rv_tetras))
# visualise_o3d_mesh(
#     mesh=mesh,
#     pcd=sparse_rv_pcd,
#     is_triangle=False,
#     show_pcd_normal=False,
#     show_voxel=True,
#     voxel_size=0.02,
#     fname="convex_rv"
# )
#
# voxelize = Voxelize(
#     width, height, depth
# )
# print("start")
# voxel_map = voxelize(rv_points, torch.from_numpy(rv_tetras).cuda().unsqueeze(0))
# print(voxel_map.shape)
# loss = torch.nn.MSELoss()
#
# # optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=5e-5)
#
# mse_loss = loss(voxel_map, torch.zeros_like(voxel_map))
# # optimizer.zero_grad()
# mse_loss.backward()
# # optimizer.step()
#
# voxel_map = voxel_map.squeeze().detach().cpu().numpy()
# xx, yy, zz = np.where(voxel_map == 1)
#
# cube = mlab.points3d(xx, yy, zz,
#              mode="cube",
#              color=(0, 1, 0),
#              scale_factor=1)
# mlab.outline()
#
# mlab.show()
