import numpy as np


voxel_width = 64
voxel_height = 64
voxel_depth = 64
cube_len = 6

label = np.zeros((voxel_width, voxel_height, voxel_depth))

# label[
# voxel_width//2 - cube_len//2:voxel_width//2 + cube_len//2,
# voxel_height//2 - cube_len//2:voxel_height//2 + cube_len//2,
# voxel_depth//2 - cube_len//2:voxel_depth//2 + cube_len//2
# ] = 1

width = 1
label[
voxel_width//2-width:voxel_width//2 + width,
voxel_height//2-width:voxel_height//2 + width,
voxel_depth//2-width:voxel_depth//2 + width
] = 1

from mayavi import mlab
xx, yy, zz = np.where(label >= 0)

cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 0, 0),
                     scale_factor=1, transparent=True, opacity=0)
mlab.outline()

xx, yy, zz = np.where(label == 1)

cube = mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 1, 0),
                     scale_factor=1)
mlab.outline()

mlab.show()

np.save("label_1.npy", label)
