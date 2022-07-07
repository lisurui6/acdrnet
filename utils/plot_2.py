# generate data
import numpy as np

x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
mx, my = np.meshgrid(x, y, indexing='ij')
mz1 = np.abs(mx) + np.abs(my)
mz2 = mx ** 2 + my ** 2

# A fix for "API 'QString' has already been set to version 1"
# see https://github.com/enthought/pyface/issues/286#issuecomment-335436808
from sys import version_info
if version_info[0] < 3:
    import pyface.qt


def v1_matplotlib():
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf1 = ax.plot_surface(mx, my, mz1, cmap='winter')
    surf2 = ax.plot_surface(mx, my, mz2, cmap='autumn')
    ax.view_init(azim=60, elev=16)
    fig.show()
    plt.show()


def v2_mayavi(transparency):
    from mayavi import mlab
    fig = mlab.figure()

    ax_ranges = [-2, 2, -2, 2, 0, 8]
    ax_scale = [1.0, 1.0, 0.4]
    ax_extent = ax_ranges * np.repeat(ax_scale, 2)

    surf3 = mlab.surf(mx, my, mz1, colormap='Blues')
    surf4 = mlab.surf(mx, my, mz2, colormap='Oranges')

    surf3.actor.actor.scale = ax_scale
    surf4.actor.actor.scale = ax_scale
    mlab.view(60, 74, 17, [-2.5, -4.6, -0.3])
    mlab.outline(surf3, color=(.7, .7, .7), extent=ax_extent)
    mlab.axes(surf3, color=(.7, .7, .7), extent=ax_extent,
              ranges=ax_ranges,
              xlabel='x', ylabel='y', zlabel='z')

    if transparency:
        surf3.actor.property.opacity = 0.5
        surf4.actor.property.opacity = 0.5
        fig.scene.renderer.use_depth_peeling = 1
    mlab.show()


v1_matplotlib()
v2_mayavi(False)
v2_mayavi(True)