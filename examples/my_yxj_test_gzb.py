from scipy.spatial.transform import Rotation as R
import numpy as np
import urdfpy

r = R.from_euler('ZYX',[3.1415927,1.5707963,2.2],degrees = False).as_matrix()
print(r)

t = urdfpy.xyz_rpy_to_matrix([-1.05, 0.5, 1.5, 3.1415927, 1.5707963, 2.2])
print(t)