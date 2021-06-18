import torch
import math
import numpy as np
from scipy.spatial import Delaunay
from shapely import geometry


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



def triangulate_within(vert, faces):
    polygon = geometry.Polygon(vert)
    output = []
    for f in range(faces.shape[0]):
        face = faces[f, :]
        triangle = geometry.Polygon(vert[face, :])
        if triangle.within(polygon):
            output.append(face)
    output = np.stack(output)
    return output
