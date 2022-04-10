from frankapy import FrankaArm
from frankapy import FrankaConstants as FC
from autolab_core import RigidTransform
import rospy

import numpy as np

if __name__ == '__main__':
    fa = FrankaArm(ros_log_level=rospy.DEBUG)

    print('Reseting Joints')
    fa.reset_joints()



    fa.reset_joints()

    # test rotation of single joints
'''
    joints = fa.get_joints()
    joints[6] += np.pi / 8
    fa.goto_joints(joints=joints, duration=5, use_impedance=True, joint_impedances=FC.DEFAULT_JOINT_IMPEDANCES)
'''

    # test rotation in end effector frame
'''
    T_ee_rot = RigidTransform(
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(45)),
        from_frame='franka_tool', to_frame='franka_tool'
    )

    T_ee_world = fa.get_pose()

    fa.goto_pose(T_ee_world * T_ee_rot)
'''

