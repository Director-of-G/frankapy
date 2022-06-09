import argparse
from frankapy import FrankaArm
import rospy

import math
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=40)
    parser.add_argument('--open_gripper', '-o', action='store_true')
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()
    if args.open_gripper:
        fa.open_gripper()
    print('Applying 0 force torque control for {}s'.format(args.time))

    q_and_manipubility_list = np.zeros((0, 8))

    start_time = time.time()

    fa.run_guide_mode(args.time,block=False)


    while not rospy.is_shutdown():
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = fa.get_joints()
        J = fa.get_jacobian(fa.get_joints())
        det = np.linalg.det(J @ J.T)
        q_and_m[0, 7] = math.sqrt(np.abs(det))
        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)

        if time.time()-start_time>=40:
            break
    
    np.save('./data/0608/q_and_m.npy', q_and_manipubility_list)
    plt.figure()
    plt.plot(q_and_manipubility_list[:, 7])
