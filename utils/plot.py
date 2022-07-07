import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA
import numpy as np
from mayavi import mlab


def equation_plane(p1, p2, p3):
    a1 = p2[0] - p1[0]
    b1 = p2[1] - p1[1]
    c1 = p2[2] - p1[2]
    a2 = p3[0] - p1[0]
    b2 = p3[1] - p1[1]
    c2 = p3[2] - p1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * p1[0] - b * p1[1] - c * p1[2])
    return a, b, c, d

image_size = 10
spatial_axes = [image_size, image_size, image_size]
filled = np.ones(spatial_axes, dtype=np.bool)

colors = np.empty(spatial_axes + [4], dtype=np.float32)
alpha = .1
colors[..., :] = [0, 0, 1, alpha]
print(colors.shape)
# colors[1] = [0, 1, 0, alpha]
# colors[2] = [0, 0, 1, alpha]
# colors[3] = [1, 1, 0, alpha]
# colors[4] = [0, 1, 1, alpha]


fig = mlab.figure()

# ax = fig.add_subplot('111', projection='3d')
# ax.voxels(filled, facecolors=colors, edgecolors='grey')

xx, yy, zz = np.where(filled == 1)
# ax = fig.add_subplot('111', projection='3d')
cube = mlab.points3d(xx, yy, zz, mode="cube", color=(0, 0, 0))
cube.actor.property.opacity = 0.001
mlab.outline()

# plot a tetrahedron

p1 = np.array([2, 2, 2])
print(p1.shape)
p2 = np.array([2, 5, 5])
p3 = np.array([8, 1, 8])
p4 = np.array([8, 8, 2])
points = np.stack([p1, p2, p3, p4], axis=0)
color = np.random.rand(3)

import matplotlib.colors as colors
#
# tri = art3d.Poly3DCollection(np.stack([p1, p2, p3], axis=0))
# tri.set_color(colors.rgb2hex(color))
# tri.set_edgecolor('k')
# ax.add_collection3d(tri)
#
# tri = art3d.Poly3DCollection(np.stack([p1, p2, p4], axis=0))
# tri.set_color(colors.rgb2hex(color))
# tri.set_edgecolor('k')
# ax.add_collection3d(tri)
#
# tri = art3d.Poly3DCollection(np.stack([p1, p3, p4], axis=0))
# tri.set_color(colors.rgb2hex(np.random.rand(3)))
# tri.set_edgecolor('k')
# ax.add_collection3d(tri)

mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
# tri = art3d.Poly3DCollection(np.stack([p2, p3, p4], axis=0))
# tri.set_color(colors.rgb2hex(color))
# tri.set_edgecolor('k')
# ax.add_collection3d(tri)

projection_color = np.random.rand(3)
tri = np.stack([p1, p3, p4], axis=0)
z_projection = tri.copy()
z_projection[:, -1] = 0
print(z_projection.shape)
mlab.triangular_mesh(z_projection[:, 0], z_projection[:, 1], z_projection[:, 2], [(0, 1, 2)])
filled = np.zeros(spatial_axes, dtype=np.bool)
filled[6, 2, 2] = 1
xx, yy, zz = np.where(filled == 1)
cube = mlab.points3d(xx, yy, zz, mode="cube", color=(1, 1, 1))


a, b, c, d = equation_plane(np.array([6, 2, 2]), p3, p4)

xx, yy = np.meshgrid(range(4, 9), range(1, 9))
z = (-d - a * xx - b * yy) / c
x1 = 5
x2 = 7
y1 = (-d - a * x1) / b
y2 = (-d - a * x2) / b



x = np.array([[p3[0], p4[0], x1, x2]])
y = np.array([[p3[1], p4[1], y1, y2]])
z = np.array([[p3[2], p4[2], 0, 0]])

# plot the plane
plane = mlab.triangular_mesh(x, y, z, [(0, 1, 2), (1, 2, 3)])
plane.actor.property.opacity = 0.1
# m.plot_surface(x, y, z, alpha=0.5)
mlab.show()

print(a, b, c, d)
# x_projection = tri.copy()
# x_projection[:, 0] = 0
#
# x_projection = art3d.Poly3DCollection(x_projection)
# x_projection.set_color(colors.rgb2hex(projection_color))
# x_projection.set_edgecolor('k')
# ax.add_collection3d(x_projection)
#
# y_projection = tri.copy()
# y_projection[:, 1] = 0
#
# y_projection = art3d.Poly3DCollection(y_projection)
# y_projection.set_color(colors.rgb2hex(projection_color))
# y_projection.set_edgecolor('k')
# ax.add_collection3d(y_projection)


plt.show()
