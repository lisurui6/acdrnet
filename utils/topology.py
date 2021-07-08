import torch
import math
import numpy as np
from scipy.spatial import Delaunay
from shapely import geometry
from matplotlib import pyplot as plt


def get_circle(batch_size, masks_size, num_points, device):
    half_dim = masks_size / 2
    half_width = half_dim
    half_height = half_dim

    # r = torch.randint(low=5, high=30, size=[1]).item()
    r = 10
    circle = []
    for x in range(0, num_points):
        circle.append([half_width + math.floor(math.cos(2 * math.pi / num_points * x) * r),
                       half_height + math.floor(math.sin(2 * math.pi / num_points * x) * r)])
    vert = np.array(circle)
    vert = (vert - half_dim) / half_dim

    tri = Delaunay(vert).simplices.copy()

    vert = torch.Tensor(vert)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face = torch.Tensor(tri)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)

    vert[:, :, :, 1] = -vert[:, :, :, 1]

    return vert, face


def get_circles(batch_size, masks_size, num_points, device):

    # mask = (B, 3, H, W)
    half_dim = masks_size / 2
    half_width = half_dim
    half_height = half_dim

    # r = torch.randint(low=5, high=30, size=[1]).item()
    r = 30
    circle1 = []
    for x in range(0, num_points):
        circle1.append([half_width + (math.cos(2 * math.pi / num_points * x) * r),
                       half_height + (math.sin(2 * math.pi / num_points * x) * r)])

    circle0 = []
    for x in range(0, num_points):
        circle0.append([half_width + (math.cos(2 * math.pi / num_points * x) * (r - 10)),
                       half_height + (math.sin(2 * math.pi / num_points * x) * (r - 10))])
    p0 = np.array(circle1[num_points//6])
    p1 = np.array(circle1[num_points//3])
    delta = r/10
    c0 = np.array([half_width, half_height])

    c2 = np.array([c0[0], c0[1] + r + delta])
    cos_theta = np.dot(p0-c2, p1-c2) / (np.linalg.norm(p0-c2) * np.linalg.norm(p1-c2))
    theta = math.acos(cos_theta) / 2

    circle2 = []
    r = np.linalg.norm(p1-c2)
    num_points2 = num_points//2
    for x in range(0, num_points2 - 1):
        circle2.append([c2[0] + (math.cos((2 * math.pi - 2 * theta) / num_points2 * (x+1) - math.pi / 2 + theta) * r),
                       c2[1] + (math.sin((2 * math.pi - 2 * theta) / num_points2 * (x+1) - math.pi / 2 + theta) * r)])
    cp_index_0 = len(circle2) - 1
    for x in range(num_points//3, num_points//6, -1):
        circle2.append(circle1[x])
    cp_index_1 = len(circle2) - 1

    vert0 = np.array(circle0)
    vert0 = (vert0 - half_dim) / half_dim
    tri0 = Delaunay(vert0).simplices.copy()
    vert0 = torch.Tensor(vert0)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face0 = torch.Tensor(tri0)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    vert0[:, :, :, 1] = -vert0[:, :, :, 1]

    circle1.extend(circle0)
    circle1.append([half_dim, half_dim])
    vert1 = np.array(circle1)
    vert1 = (vert1 - half_dim) / half_dim
    tri1 = Delaunay(vert1).simplices.copy()
    mask = ~(tri1 == vert1.shape[0] - 1).any(axis=1)
    tri1 = tri1[mask, :]
    vert1 = vert1[:-1, :]

    vert1 = torch.Tensor(vert1)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face1 = torch.Tensor(tri1)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    vert1[:, :, :, 1] = -vert1[:, :, :, 1]

    vert2 = np.array(circle2)
    vert2 = (vert2 - half_dim) / half_dim
    tri2 = Delaunay(vert2).simplices.copy()
    tri2 = triangulate_within(vert2, tri2)
    vert2 = torch.Tensor(vert2)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face2 = torch.Tensor(tri2)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    vert2[:, :, :, 1] = -vert2[:, :, :, 1]

    return vert0, face0, vert1, face1, vert2, face2, cp_index_0, cp_index_1



def get_circles_2(batch_size, masks_size, num_points, device):

    # mask = (B, 3, H, W)
    half_dim = masks_size / 2
    half_width = half_dim
    half_height = half_dim

    # r = torch.randint(low=5, high=30, size=[1]).item()
    r = 30
    circle1 = []
    for x in range(0, num_points):
        circle1.append([half_width + (math.cos(2 * math.pi / num_points * x) * r),
                       half_height + (math.sin(2 * math.pi / num_points * x) * r)])
    circle1.append([half_width, half_height])
    circle0 = []
    for x in range(0, num_points):
        circle0.append([half_width + (math.cos(2 * math.pi / num_points * x) * (r - 10)),
                       half_height + (math.sin(2 * math.pi / num_points * x) * (r - 10))])
    circle0.append([half_width, half_height])
    p0 = np.array(circle1[num_points//6])
    p1 = np.array(circle1[num_points//3])
    delta = r/10
    c0 = np.array([half_width, half_height])

    c2 = np.array([c0[0], c0[1] + r + delta])
    cos_theta = np.dot(p0-c2, p1-c2) / (np.linalg.norm(p0-c2) * np.linalg.norm(p1-c2))
    theta = math.acos(cos_theta) / 2

    circle2 = []
    r = np.linalg.norm(p1-c2)
    num_points2 = num_points//2
    for x in range(0, num_points2 - 1):
        circle2.append([c2[0] + (math.cos((2 * math.pi - 2 * theta) / num_points2 * (x+1) - math.pi / 2 + theta) * r),
                       c2[1] + (math.sin((2 * math.pi - 2 * theta) / num_points2 * (x+1) - math.pi / 2 + theta) * r)])
    cp_index_0 = len(circle2) - 1
    for x in range(num_points//3, num_points//6, -1):
        circle2.append(circle1[x])
    cp_index_1 = len(circle2) - 1

    vert0 = np.array(circle0)
    vert0 = (vert0 - half_dim) / half_dim
    tri0 = Delaunay(vert0).simplices.copy()
    vert0 = torch.Tensor(vert0)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face0 = torch.Tensor(tri0)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    vert0[:, :, :, 1] = -vert0[:, :, :, 1]

    vert1 = np.array(circle1)
    vert1 = (vert1 - half_dim) / half_dim
    tri1 = Delaunay(vert1).simplices.copy()
    vert1 = torch.Tensor(vert1)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face1 = torch.Tensor(tri1)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    vert1[:, :, :, 1] = -vert1[:, :, :, 1]

    vert2 = np.array(circle2)
    vert2 = (vert2 - half_dim) / half_dim
    tri2 = Delaunay(vert2).simplices.copy()
    tri2 = triangulate_within(vert2, tri2)
    vert2 = torch.Tensor(vert2)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1)
    face2 = torch.Tensor(tri2)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    vert2[:, :, :, 1] = -vert2[:, :, :, 1]

    return vert0, face0, vert1, face1, vert2, face2, cp_index_0, cp_index_1


def get_circles_3(par1, par2, batch_size, masks_size, num_points, device, tri0=None, tri1=None, tri2=None):
    """
        par1 (B, 2): c0(x, y)
        par2 (B, 5): r1, factor 0 (r0/r1), theta2/theta2_max, d_c2_c0, theta_c2

        theta2_max = pi if c2 inside c1, otherwise arcsin(r0/d_c2_c0)
    """
    # mask = (B, 3, H, W)
    half_dim = masks_size / 2
    # vectorize circle 0 and circle 1
    c0x = (par1[:, 0] / 2 + 0.5) * 127
    c0y = (par1[:, 1] / 2 + 0.5) * 127
    r1 = par2[:, 0] * 128

    # c0x = (torch.tensor(0).float().repeat(batch_size).cuda() / 2 + 0.5) * 127
    # c0y = (torch.tensor(0).float().repeat(batch_size).cuda() / 2 + 0.5) * 127
    # r1 = torch.tensor(30/128).float().repeat(batch_size).cuda() * 128
    r0 = torch.mul(par2[:, 1], r1).repeat(num_points, 1).transpose(1, 0)

    c0_phase = torch.arange(num_points).repeat(batch_size, 1).cuda()
    c0_phase = 2 * math.pi * c0_phase / num_points
    z_c0 = torch.complex(real=c0x, imag=c0y)
    c0_angle = torch.exp(torch.complex(real=torch.tensor(0).float().repeat(batch_size, num_points).cuda(), imag=c0_phase))
    circle0 = z_c0.repeat(num_points, 1).transpose(1, 0) + r0 * c0_angle
    circle0 = torch.cat([circle0, z_c0.unsqueeze(1)], dim=1)
    circle0 = torch.view_as_real(circle0)

    vert0 = (circle0 - half_dim) / half_dim
    if tri0 is None:
        tri0 = Delaunay(vert0[0].detach().cpu().numpy()).simplices.copy()
    vert0 = vert0.unsqueeze(1)
    vert0[:, :, :, 1] = -vert0[:, :, :, 1]
    face0 = torch.Tensor(tri0)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)

    circle1 = z_c0.repeat(num_points, 1).transpose(1, 0) + r1.repeat(num_points, 1).transpose(1, 0) * c0_angle
    circle1 = torch.cat([circle1, z_c0.unsqueeze(1)], dim=1)
    circle1 = torch.view_as_real(circle1)

    vert1 = (circle1 - half_dim) / half_dim
    vert1 = vert1.unsqueeze(1)
    vert1[:, :, :, 1] = -vert1[:, :, :, 1]
    face1 = face0

    # to compute circle 2

    dmin = r1 * 1/2
    dmax = r1 * 3/2
    d_c2_c0 = par2[:, 3] * (dmax - dmin) + dmin
    theta_c2 = par2[:, 4] * math.pi * 2
    # theta_c2 = torch.tensor(math.pi / 2).float().repeat(batch_size).cuda()

    z_c2 = z_c0 + d_c2_c0 * torch.exp(torch.complex(real=torch.tensor(0).float().cuda().repeat(batch_size), imag=theta_c2.float()))
    theta2_max = torch.tensor(math.pi * 3 / 4).float().repeat(batch_size).cuda()
    theta2_min = torch.tensor(math.pi * 1 / 6).float().repeat(batch_size).cuda()

    theta2 = theta2_min + par2[:, 2] * (theta2_max - theta2_min)
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
    arc_phase = theta_p0 + torch.mul(theta_p1 - theta_p0, arc_count) / n_arc_points
    arc_angle = torch.exp(torch.complex(real=torch.tensor(0).float().repeat(batch_size, n_arc_points).cuda(), imag=arc_phase))
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

    arc_phase = theta_c2_p0 + torch.mul(theta_c2_p1 - theta_c2_p0, arc_count) / n_arc_points
    arc_angle = torch.exp(torch.complex(real=torch.tensor(0).float().repeat(batch_size, n_arc_points).cuda(), imag=arc_phase))
    arc_2 = z_c2.repeat(n_arc_points, 1).transpose(1, 0) + r2.repeat(n_arc_points, 1).transpose(1, 0) * arc_angle

    circle2 = torch.cat([torch.view_as_real(arc_2), torch.view_as_real(arc_1)], dim=1)
    vert2 = (circle2 - half_dim) / half_dim
    if tri2 is None:
        tri2 = Delaunay(vert2[0].detach().cpu().numpy()).simplices.copy()
        tri2 = triangulate_within(vert2[0].detach().cpu().numpy(), tri2)
    vert2 = vert2.unsqueeze(1)
    vert2[:, :, :, 1] = -vert2[:, :, :, 1]
    face2 = torch.Tensor(tri2)[None, None, ...].to(device).repeat(batch_size, 1, 1, 1).type(torch.int32)
    return vert0, face0, vert1, face1, vert2, face2, tri0, tri1, tri2


def triangulate_circle_2(n_c2, n_tot):
    print(n_c2, n_tot)
    tris = []
    if n_c2 > n_tot / 2:
        pass
    for i in range(n_c2):
        tris.extend([
            [i, i+1, n_tot - i - 1],
            [i, i+1, n_tot - i - 2],
            [n_tot - i-1, n_tot - i - 2, i],
            [n_tot - i-1, n_tot - i - 2, i + 1],
        ])
    return np.array(tris)


def pad_tri2(tri2, n):
    padding = []
    for i in range(n - tri2.shape[0]):
        padding.append(
            [i, i + 1, i + 2]
        )
    padding = np.array(padding)
    return np.concatenate([tri2, padding], axis=0)


def triangulate_within(vert, faces):
    polygon = geometry.Polygon(vert)
    output = []
    for f in range(faces.shape[0]):
        face = faces[f, :]
        triangle = geometry.Polygon(vert[face, :])
        if triangle.within(polygon):
            output.append(face)
    if len(output) == 0:
        vert = vert * 64 + 64
        plt.imshow(np.zeros((128, 128)))
        plt.plot(vert[:, 0], vert[:, 1], 'bx-')
        for f in range(faces.shape[0]):
            p1 = vert[faces[f, 0], :]
            p2 = vert[faces[f, 1], :]
            p3 = vert[faces[f, 2], :]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
            plt.plot([p1[0], p3[0]], [p1[1], p3[1]], 'k-')
            plt.plot([p3[0], p2[0]], [p3[1], p2[1]], 'k-')

        plt.show()
    output = np.stack(output)
    return output
