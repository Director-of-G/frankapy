"""
    my_adaptive_control.py, 2022-05-01
    Copyright 2022 IRM Lab. All rights reserved.
"""
from tkinter import N, W
from frankapy.franka_arm import FrankaArm
import rospy
import tf

from scipy.spatial.transform import Rotation as R
import numpy as np
import math

# Definition of Constants
class MyConstants(object):
    """
        @ Class: MyConstants
        @ Function: get all the constants in this file
    """
    FX_HAT = 2337.218017578125
    FY_HAT = 2341.164794921875
    U0 = 746.3118044533257
    V0 = 564.2590475570069

class MyJacobianHandler(object):
    """
        @ Class: MyJacobianHandler
        @ Function: get Jacobian of 6*1 velocity vector between different frames in /tf
    """
    def __init__(self) -> None:
        self.tf_listener = tf.TransformListener()

    def calcJacobian(self, from_frame=None, to_frame=None):
        frameA = 'world' if from_frame is None else from_frame
        frameB = 'world' if to_frame is None else to_frame
        try:
            (trans_AinB, rot_A2B) = self.tf_listener.lookupTransform(target_frame=frameA, source_frame=frameB, time=rospy.Time(0))
            trans_BinA = (-trans_AinB).reshape(3, 1)
            (px, py, pz) = np.matmul(rot_A2B, trans_BinA)
            P_cross_mat = np.array([[0, -pz, py], \
                                    [pz, 0, -px], \
                                    [-py, px, 0]])
            J_A2B = np.block([[rot_A2B, P_cross_mat], \
                              [np.zeros((3, 3)), rot_A2B]])

            return J_A2B
        except:
            raise NotImplementedError('Failed to get Jacobian matrix from %s to %s' % (from_frame, to_frame))

class RegionFunction(object):
    class ImageSpaceRegionFunction(object):
        def __init__(self, x_d, b, Kv) -> None:
            self.x_d = x_d
            self.b = b
            self.Kv = Kv
        def set_x_d(self, x_d):
            self.x_d = x_d
        def set_b(self, b):
            self.b = b
        def set_Kv(self, Kv):
            self.Kv = Kv
        def fv(self, x):
            return np.linalg.norm((x - self.x_d) / self.b, ord=2)
        def in_region(self, x):
            fv = self.fv(x)
            return fv <= 0
        def Pv(self, x):
            fv = self.fv(x)
            return 0.5 * self.Kv * (1 - math.min(0, fv) ** 2)
        def kesi_x(self, x):
            partial_fv = 2 * (x - self.x_d) / (self.b ** 2)
            partial_fv = partial_fv.reshape(-1, 1)
            return - self.Kv * self.fv(x) * partial_fv

    class CartesianSpaceRegionFunction(object):
        def __init__(self, r_c=None, c=None, Kc=None) -> None:
            """
                r_c is the desired Cartesian configuration, which is [x, y, z, r, p, y].
                r∈[-pi, pi], p∈[-pi/2, pi/2], y∈[-pi, pi], which is euler angle in the order of 'XYZ'
            """
            self.r_c = r_c
            self.c = c
            self.Kc = Kc

        def set_r_c(self, r_c):
            self.r_c = r_c

        def set_r_c_with_pose(self, pose):
            """
                Given the pose in Cartesian space, which is a RigidTransform(translation, rotation).
                Calculate the corresponding r_c, which is [x, y, z, r, p, y].
                Avoid assigning y too close to pi/2 and -pi/2
            """
            x, y, z = pose.translation
            r, p, y = R.from_matrix(pose.rotation).as_euler(seq='XYZ', degrees=False)
            r_c = np.array([x, y, z, r, p, y])
            self.set_r_c(r_c)

        def set_c(self, c):
            self.c = c
            
        def set_Kc(self, Kc):
            self.Kc = Kc

        def fc(self, r):
            """
                r should have the size of (6,)
            """
            r, r_c, c = r.reshape(6, 1), self.r_c.reshape(6, 1), self.c.reshape(6, 1)
            roll, yaw = r[3], r[5]
            if abs(roll - r_c[3]) > np.pi:
                roll = 2 * r_c[3] - roll
            if abs(yaw - r_c[5]) > np.pi:
                yaw = 2 * r_c[5] - yaw
            r[3], r[5] = roll, yaw
            fc = ((r - r_c) / c) ** 2 - 1
            return fc

        def in_region(self, r):
            fc = self.fc(r)
            return (fc <= 0).reshape(6, 1)

        def Pc(self, r):
            fc = self.fc(r)
            return np.sum(0.5 * self.Kc * np.max(0, fc) ** 2)

        def kesi_r(self, r):
            r, r_c, c = r.reshape(6, 1), self.r_c.reshape(6, 1), self.c.reshape(6, 1)
            roll, yaw = r[3], r[5]
            if abs(roll - r_c[3]) > np.pi:
                roll = 2 * r_c[3] - roll
            if abs(yaw - r_c[5]) > np.pi:
                yaw = 2 * r_c[5] - yaw
            r[3], r[5] = roll, yaw
            partial_fc = 2 * (r - r_c) / (c ** 2)
            partial_fc = partial_fc.reshape(-1, 1)
            return self.Kc * np.max(0, self.fc(r)) * partial_fc

    class JointSpaceRegionFunction(object):
        # @TODO
        def __init__(self) -> None:
            pass
        def kesi_q(self, q):
            pass


class AdaptiveImageJacobian(object):
    """
        @ Class: AdaptiveImageJacobian
        @ Function: adaptive update the image jacobian
    """
    def __init__(self, fa: FrankaArm =None, n_k=3, Js=None, x=None, L=None, W_hat=None) -> None:
        if fa is None:
            raise ValueError('FrankaArm handle is not provided!')
        self.fa = fa
        self.n_k = n_k

        # Js here transforms Cartesian space velocity (v, w): 6 * 1 to image space velocity (du, dv): 2 * 1
        # Js is 2 * 6
        if Js is not None:
            if Js.shape != (2, 6):
                raise ValueError('Dimension of Js should be (2, 6), not ' + str(Js.shape) + '!')
            self.Js_hat = Js
        else:  # control with precise Js
            if x is None:
                raise ValueError('Target point x on the image plane should not be empty!')
            fx, fy = MyConstants.FX_HAT, MyConstants.FY_HAT
            u0, v0 = MyConstants.U0, MyConstants.V0
            u, v = x[0] - u0, x[1] - v0
            z = 1
            J_cam2img = np.array([[fx/z, 0, -u/z, -u*v/fx, (fx+u**2)/fx, -v], \
                                  [0, fy/z, -v/z, -(fy+v**2)/fy, u*v/fy, u]])
            J_base2cam = MyJacobianHandler.calcJacobian(from_frame='panda_link0', to_frame='camera_link')
            
            rot_ee = fa.get_pose().rotation  # (rotation matrix of the end effector)
            (r, p, y) = R.from_matrix(rot_ee).as_euler('XYZ', degrees=False)  # @TODO: intrinsic rotation, first 'X', second 'Y', third'Z', to be checked
            J_baserpy2w = np.block([[np.eye(3), np.zeros((3, 3))], \
                                    [np.zeros((3, 3)), np.array([[1, 0, math.sin(p)], \
                                                                 [0, math.cos(r), -math.cos(p) * math.sin(r)], \
                                                                 [0, math.sin(r), math.cos(p) * math.cos(r)]])]])
            self.Js_hat = J_cam2img @ J_base2cam  # Js_hat = J_base2img

        if L is not None:
            if L.shape != (n_k, n_k):
                raise ValueError('Dimension of L should be ' + str((n_k, n_k)) + '!')
            self.L = L
        else:
            raise ValueError('Matrix L should not be empty!')

        if W_hat is not None:
            if W_hat.shape != (12, n_k):
                raise ValueError('Dimension of W_hat should be ' + str((12, n_k)) + '!')
            self.W_hat = W_hat
        else:
            raise ValueError('Matrix W_hat should not be empty!')

        # @TODO
        self.theta = self.get_theta(n_k, r)
        self.image_space_region_controller = None
        self.cartesian_space_region_controller = None
        self.joint_space_region_controller = None

    def get_theta(self, n_k, r):
        # @TODO
        if r is not None:
            return np.random.rand(n_k, 1)
        else:
            raise ValueError('Cartesian configuration r should not be empty!')

    def set_region_controller(self, image_space_controller, cartesian_space_controller, joint_space_controller):
        self.image_space_region_controller = image_space_controller
        self.cartesian_space_region_controller = cartesian_space_controller
        self.joint_space_controller = joint_space_controller

    def get_kesi_x(self, x):
        return self.image_space_region_controller.kesi_x(x)

    def get_kesi_r(self, r):
        return self.cartesian_space_region_controller.kesi_r(r)

    def get_kesi_q(self, q):
        return self.joint_space_region_controller(q)

    def get_Js_hat(self):
        return self.Js_hat

    def update(self):
        r = self.fa.get_pose()  # get r: including translation and rotation matrix
        q = self.fa.get_joints()  # get q: joint angles
        theta = self.get_theta(self.n_k, r)  # get the neuron values theta(r)
        J = self.fa.get_jacobian(q)  # get the analytic jacobian (6*7)
        J_inv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        
        kesi_x = self.get_kesi_x()
        kesi_r = self.get_kesi_r()
        kesi_q = self.get_kesi_q()
        dW_hat1_T = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_r + J_inv.T @ kesi_q).T * kesi_x[0]
        dW_hat2_T = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_r + J_inv.T @ kesi_q).T * kesi_x[1]

        dW_hat1, dW_hat2 = dW_hat1_T.T, dW_hat2_T.T  # (6, n_k), (6, n_k)
        dW_hat = np.concatenate((dW_hat1, dW_hat2), axis=0)  # (12, n_k)
        self.W_hat = self.W_hat + dW_hat  # (12, n_k)

        Js_hat1 = self.W_hat[:6, :] @ theta  # (6, 1)
        Js_hat2 = self.W_hat[6:, :] @ theta  # (6, 1)
        Js_hat = np.concatenate((Js_hat1.reshape(1, -1), Js_hat2.reshape(1, -1)), axis=0)  # (2, 6)
        self.Js_hat = Js_hat
