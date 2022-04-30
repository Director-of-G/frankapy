from frankapy import FrankaArm

if __name__ == '__main__':
    fa = FrankaArm()
    joints = fa.get_joints()
    fa.open_gripper()
    print('joint angles: ', joints)
