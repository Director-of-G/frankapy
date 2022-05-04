from frankapy import FrankaArm
import numpy as np

fa = FrankaArm()
joint = fa.get_joints()
J = fa.get_jacobian_joint4(joint)
print(J)
print(J@np.array([[0.1],[0],[0],[0]]))