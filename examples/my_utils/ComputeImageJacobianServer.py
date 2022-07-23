import sys
sys.path.append('/home/roboticslab/yxj/frankapy/catkin_ws/src')
from franka_interface_msgs.srv import ComputeImageJacobianUpdate, ComputeImageJacobianUpdateResponse

import rospy

import numpy as np
import math

import time

def handle_update_image_jacobian(req):
    L = np.array(req.L).flatten()
    n_k = int(math.sqrt(L.shape[0]))
    L = L.reshape(n_k, n_k)
    theta = np.array(req.theta).reshape(n_k, 1)
    Js_hat = np.array(req.Js_hat).reshape(2, 6)
    kesi_x = np.array(req.kesi_x).reshape(2, 1)
    kesi_rall = np.array(req.kesi_rall).reshape(6, 1)
    J_pinv = np.array(req.J_pinv).reshape(7, 6)
    kesi_q = np.array(req.kesi_q).reshape(7, 1)
    kesi_x_prime = np.array(req.kesi_x_prime).reshape(6, 12)
    
    compute_start_T = time.time()
    dW_hat = - L @ theta @ (Js_hat.T @ kesi_x + kesi_rall + J_pinv.T @ kesi_q).T
    dW_hat = dW_hat @ kesi_x_prime
    compute_T = time.time() - compute_start_T

    return ComputeImageJacobianUpdateResponse(dW_hat.flatten(), compute_T)

def compute_image_jacobian_server():
    rospy.init_node('compute_image_jacobian_server')
    s = rospy.Service('compute_image_jacobian_update', ComputeImageJacobianUpdate, handle_update_image_jacobian)
    print("Ready to compute image jacobian.")
    rospy.spin()

if __name__ == "__main__":
    compute_image_jacobian_server()
