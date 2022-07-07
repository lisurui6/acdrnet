import torch
import numpy as np
from rasterizor.voxelize import Voxelize
from rasterizor.utils import is_point_in_tetrahedron, is_point_in_triangle, line_intersect_plane, is_projected_point_in_triangle

import nibabel as nib
from typing import Tuple
from scipy.ndimage import zoom
from torch.autograd import Variable


voxel_width = 64
voxel_height = 64
voxel_depth = 64


def run_test_if_point_in_tetra():
    v1 = [voxel_width / 2, voxel_depth / 2, voxel_height]
    v2 = [0, voxel_depth, 0]
    v3 = [voxel_width / 2, 0, 0]
    v4 = [voxel_width, voxel_depth, 0]
    point = [0, 0, 0]
    assert not is_point_in_tetrahedron(point, v1, v2, v3, v4)
    point = [voxel_width / 2, voxel_depth / 2, voxel_height / 2]
    assert is_point_in_tetrahedron(point, v1, v2, v3, v4)


def run_test_if_point_in_triangle():
    v1 = [0, 0]
    v2 = [10, 0]
    v3 = [5, 10]
    point = [0, 10]
    assert not is_point_in_triangle(point, v1, v2, v3)
    point = [5, 5]
    assert is_point_in_triangle(point, v1, v2, v3)


def run_test_line_intersect_plane():
    line_point = [5., 5., 10.]
    line_direction = [0, 0, 1.]
    p1 = [0, 0, 4.]
    p2 = [10., 0, 4.]
    p3 = [5., 10., 4.]
    print(line_intersect_plane(line_point, line_direction, p1, p2, p3))

    line_point = [1., 3.54, 0.]
    line_direction = [0, 0, 1.]
    p1 = [0, 0, 0.]
    p2 = [10., 10, 0.]
    p3 = [0., 10., 10.]
    print(line_intersect_plane(line_point, line_direction, p1, p2, p3))


def run_test_is_projected_point_in_triangle():
    v1 = [0, 0, 3]
    v2 = [10, 0, 3]
    v3 = [5, 10, 3]
    point = [0, 10, 10]
    assert not (is_projected_point_in_triangle(point, v1, v2, v3))
    point = [5, 5, 10]
    assert (is_projected_point_in_triangle(point, v1, v2, v3))

    v1 = [5, 5, 0]
    v2 = [0, 5, 5]
    v3 = [0, 0, 0]
    point = [0, 0, 5]
    assert not (is_projected_point_in_triangle(point, v1, v2, v3))
    point = [2, 0, 2]
    assert (is_projected_point_in_triangle(point, v1, v2, v3))

    v1 = [0.5, 0, -1]
    v2 = [0.5, 0, 1]
    v3 = [0.92, 1.2, 0]
    point = [1, 1.4, 0]
    assert not (is_projected_point_in_triangle(point, v1, v2, v3))


run_test_if_point_in_tetra()
run_test_if_point_in_triangle()
run_test_line_intersect_plane()
run_test_is_projected_point_in_triangle()