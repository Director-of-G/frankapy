import sys
import rospy
sys.path.append('/home/roboticslab/yxj/frankapy/catkin_ws/src')
from franka_interface_msgs.srv import ComputeImageJacobianUpdate

import numpy as np
import time

import pdb

def handle_update_image_client(L, theta, Js_hat, kesi_x, kesi_rall, J_pinv, kesi_q, kesi_x_prime):
    try:
        compute_image_jacobian = rospy.ServiceProxy('compute_image_jacobian_update', ComputeImageJacobianUpdate)
        L = L.flatten()
        theta = theta.flatten()
        Js_hat = Js_hat.flatten()
        kesi_x = kesi_x.flatten()
        kesi_rall = kesi_rall.flatten()
        J_pinv = J_pinv.flatten()
        kesi_q = kesi_q.flatten()
        kesi_x_prime = kesi_x_prime.flatten()
        resp = compute_image_jacobian(L, theta, Js_hat, kesi_x, kesi_rall, J_pinv, kesi_q, kesi_x_prime)
        print(np.array(resp.dW_hat).shape)
        return resp.dW_hat, float(resp.time)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    L = np.eye(1000)
    theta = np.random.rand(1000, 1)
    Js_hat = np.random.rand(2, 6)
    kesi_x = np.random.rand(2, 1)
    kesi_rall = np.random.rand(6, 1)
    kesi_q = np.random.rand(7, 1)
    J_pinv = np.random.rand(7, 6)
    kesi_x_prime = np.random.rand(6, 12)

    rospy.wait_for_service('compute_image_jacobian_update')

    total_T = 0
    n_iters = 10
    for iter in range(n_iters):
        _, compute_T = handle_update_image_client(L, theta, Js_hat, kesi_x, kesi_rall, J_pinv, kesi_q, kesi_x_prime)
        total_T += compute_T
    print('time per iter: %.5f' % (total_T / n_iters))
