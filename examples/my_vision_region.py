"""
    my_adaptive_control.py, 2022-05-01
    Copyright 2022 IRM Lab. All rights reserved.
"""
from cProfile import label
import collections
from turtle import position
from matplotlib.contour import ContourLabeler
# from frankapy.franka_arm import FrankaArm
import rospy
# import tf

from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import time
import pdb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from my_utils import Quat, RadialBF
from matplotlib import pyplot as plt
from std_msgs.msg import Float64MultiArray
from matplotlib import pyplot as plt
import pickle

from franka_example_controllers.msg import JointVelocityCommand
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm,SensorDataMessageType
from frankapy import FrankaConstants as FC

from geometry_msgs.msg import PointStamped

import os, sys
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
    CARTESIAN_CENTER = np.array([-0.0068108842682527, 0.611158320250102, 0.1342875493162069])

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

class ImageSpaceRegion(object):
    def __init__(self, x_d=None, b=None, Kv=None) -> None:
        self.x_d = x_d  # (1, 2)
        self.b = b  # (1, 2)
        self.Kv = Kv  # float
    def set_x_d(self, x_d):
        self.x_d = x_d
    def set_b(self, b):
        self.b = b
    def set_Kv(self, Kv):
        self.Kv = Kv
    def fv(self, x):  # x => (1, 2)
        return np.linalg.norm((x - self.x_d) / self.b, ord=2) - 1
    def in_region(self, x):  # x => (1, 2)
        fv = self.fv(x)
        return (fv <= 0)
    def Pv(self, x):  # x => (1, 2)
        fv = self.fv(x)
        return 0.5 * self.Kv * (1 - np.minimum(0, fv) ** 2)
    def kesi_x(self, x):  # x => (1, 2)
        partial_fv = 2 * (x - self.x_d) / (self.b ** 2)
        partial_fv = partial_fv.reshape(1, -1)
        return - self.Kv * np.minimum(0, self.fv(x)) * partial_fv

class CartesianSpaceRegion(object):
    def __init__(self, r_c=None, c=None, Kc=None) -> None:
        """
            r_c is the desired Cartesian configuration, which is [x, y, z, r, p, y].
            r???[-pi, pi], p???[-pi/2, pi/2], y???[-pi, pi], which is euler angle in the order of 'XYZ'
        """
        self.r_c = r_c  # (1, 3)
        self.c = c  # (1, 3)
        self.Kc = Kc  # (1, 3)

    def set_r_c(self, r_c):
        self.r_c = r_c

    def set_r_c_with_pose(self, pose):  # deprecated
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

    def fc(self, r, with_rot=False):  # r => (1, 3)
        """
            r should have the size of (6,)
        """
        if with_rot:  # deprecated
            r, r_c, c = r.reshape(6, 1), self.r_c.reshape(6, 1), self.c.reshape(6, 1)
            roll, yaw = r[3], r[5]
            if abs(roll - r_c[3]) > np.pi:
                roll = 2 * r_c[3] - roll
            if abs(yaw - r_c[5]) > np.pi:
                yaw = 2 * r_c[5] - yaw
            r[3], r[5] = roll, yaw
        else:
            r_c = self.r_c
            c = self.c
        fc = ((r - r_c) / c) ** 2 - 1
        return fc

    def in_region(self, r):
        fc = self.fc(r)
        return (fc <= 0).reshape(1, -1)

    def Pc(self, r):
        fc = self.fc(r)
        return np.sum(0.5 * self.Kc * np.maximum(0, fc) ** 2)

    def kesi_r(self, r, with_rot=False):  # r => (1, 3)
        if with_rot:  # deprecated
            r, r_c, c = r.reshape(6, 1), self.r_c.reshape(6, 1), self.c.reshape(6, 1)
            roll, yaw = r[3], r[5]
            if abs(roll - r_c[3]) > np.pi:
                roll = 2 * r_c[3] - roll
            if abs(yaw - r_c[5]) > np.pi:
                yaw = 2 * r_c[5] - yaw
            r[3], r[5] = roll, yaw
        else:
            r_c = self.r_c
            c = self.c
        partial_fc = 2 * (r - r_c) / (c ** 2)
        partial_fc = partial_fc.reshape(1, -1)
        return (self.Kc * np.maximum(0, self.fc(r)) * partial_fc).reshape(1, -1)

class CartesianQuatSpaceRegion(object):
    def __init__(self, q_g:np.ndarray=None, Ko=1) -> None:
        self.q_g = Quat(q_g)  # Quat
        self.q_diff = Quat()  # Quat
        self.Ko = Ko  # float

    def set_q_g(self, q_g: np.ndarray):
        self.q_g = Quat(q_g)  # Quat

    def set_Ko(self, Ko):
        self.Ko = Ko  # float
    
    def fo(self, q:Quat, return_diff=False):
        q_unit = q.unit_()
        q_g_unit = self.q_g.unit_()
        self.q_diff = q_unit.dq_(q_g_unit).unit_()
        # self.q_diff = q_g_unit.dq_(q_unit)
        if return_diff:
            return self.Ko * self.q_diff.logarithm_(return_norm=True) - 1, self.q_diff
        else:
            return self.Ko * self.q_diff.logarithm_(return_norm=True) - 1

    def in_region(self, q:Quat):
        fo = self.fo(q)
        return (fo <= 0)

    def Po(self, q:Quat):
        fo = self.fo(q)

        return 0.5 * (np.maximum(0, fo)) ** 2

    # deprecated
    def get_Jrot(self, q:Quat):
        r_o = q.angle_axis_()
        norm_r_o = np.linalg.norm(r_o, ord=2)
        J_rot = np.zeros((4, 3))
        J_rot[0, :] = - r_o * math.sin(norm_r_o / 2) / (2 * norm_r_o)
        J_rot_matBC = [[[r_o[0]**2, r_o[1]**2 + r_o[2]**2], [r_o[0]*r_o[1], -r_o[0]*r_o[1]], [r_o[0]*r_o[2], -r_o[0]*r_o[2]]], \
                       [[r_o[1]*r_o[0], -r_o[1]*r_o[0]], [r_o[1]**2, r_o[2]**2 + r_o[0]**2], [r_o[1]*r_o[2], -r_o[1]*r_o[2]]], \
                       [[r_o[2]*r_o[0], -r_o[2]*r_o[0]], [r_o[2]*r_o[1], -r_o[2]*r_o[1]], [r_o[2]**2, r_o[0]**2 + r_o[1]**2]]]
        J_rot_matBC = np.dot(np.array(J_rot_matBC), np.array([math.acos(norm_r_o/2)/(2*norm_r_o**2), math.sin(norm_r_o/2)/(norm_r_o**3)]))
        pass

    def kesi_rq(self, q:np.ndarray):  # q => (4,)
        q, q_g = Quat(q).unit_(), self.q_g.unit_()
        q_o, q_g_inv = q.quat, q_g.inverse_().quat
        fo, q_diff = self.fo(q, return_diff=True)
        q_sign = np.array([-1, 1, 1])
        partial_v_q = np.array([q_g_inv[0] + q_o[0] * np.sum(q_g_inv[[1, 2, 3]]/q_o[[1, 2, 3]]), \
                                -q_g_inv[1] + q_o[1] * np.sum(q_g_inv[[0, 2, 3]]/q_o[[0, 2, 3]]*q_sign), \
                                -q_g_inv[2] + q_o[2] * np.sum(q_g_inv[[0, 1, 3]]/q_o[[0, 1, 3]]*q_sign), \
                                -q_g_inv[3] + q_o[3] * np.sum(q_g_inv[[0, 1, 2]]/q_o[[0, 1, 2]]*q_sign)])  # (4,)
        u = q_diff.u_()
        norm_u = np.linalg.norm(u, ord=2)

        partial_P_q = (-(self.Ko / norm_u) * np.maximum(0, fo) * partial_v_q * (norm_u > 0)).reshape(1, 4)
        J_rot = q.jacobian_rel2_axis_angle_()  # (4, 3)

        return (partial_P_q @ J_rot).reshape(1, -1)

    def kesi_rq_omega(self, q:np.ndarray):
        q, q_g = Quat(q).unit_(), self.q_g.unit_()
        q_diff = q.dq_(q_g)

        axis, theta = q_diff.axis_angle_(split=True)
        axis_normalized = axis / np.linalg.norm(axis, ord=2)
        if theta > math.pi:
            theta = - (2 * math.pi - theta)
        return theta * axis_normalized.reshape(1, 3)
        

class JointSpaceRegion(object):
    def __init__(self) -> None:
        self.single_kq = np.zeros((1, 0))
        self.single_kr = np.zeros((1, 0))
        self.single_qc = np.zeros((1, 0))
        self.single_scale = np.zeros((1, 0))
        self.single_mask = np.zeros((1, 0))
        self.single_qbound = np.zeros((1, 0))
        self.single_qrbound = np.zeros((1, 0))
        self.single_inout = np.zeros((1, 0))

        self.multi_kq = np.zeros((1, 0))
        self.multi_kr = np.zeros((1, 0))
        self.multi_qc = np.zeros((7, 0))
        self.multi_scale = np.zeros((7, 0))
        self.multi_mask = np.zeros((7, 0))
        self.multi_qbound = np.zeros((1, 0))
        self.multi_qrbound = np.zeros((1, 0))
        self.multi_inout = np.zeros((1, 0), dtype=np.bool8)  # in => 1, out => (-1)

    def add_region_single(self, qc, qbound, qrbound, mask, kq=1, kr=1, inner=False, scale=1):
        self.single_kq = np.concatenate((self.single_kq, [[kq]]), axis=1)
        self.single_kr = np.concatenate((self.single_kr, [[kr]]), axis=1)
        self.single_qc = np.concatenate((self.single_qc, [[qc]]), axis=1)
        self.single_scale = np.concatenate((self.single_scale, [[scale]]), axis=1)
        self.single_mask = np.concatenate((self.single_mask, [[mask]]), axis=1)
        self.single_qbound = np.concatenate((self.single_qbound, [[qbound]]), axis=1)
        self.single_qrbound = np.concatenate((self.single_qrbound, [[qrbound]]), axis=1)
        self.single_inout = np.concatenate((self.single_inout, [[int(inner) * 2 - 1]]), axis=1)

    def add_region_multi(self, qc, qbound, qrbound, mask, kq=1, kr=1, inner=False, scale=np.ones((7, 1))):
        self.multi_kq = np.concatenate((self.multi_kq, [[kq]]), axis=1)
        self.multi_kr = np.concatenate((self.multi_kr, [[kr]]), axis=1)
        self.multi_qc = np.concatenate((self.multi_qc, qc.reshape(7, 1)), axis=1)
        self.multi_scale = np.concatenate((self.multi_scale, scale.reshape(7, 1)), axis=1)
        self.multi_mask = np.concatenate((self.multi_mask, mask.reshape(7, 1)), axis=1)
        self.multi_qbound = np.concatenate((self.multi_qbound, [[qbound]]), axis=1)
        self.multi_qrbound = np.concatenate((self.multi_qrbound, [[qrbound]]), axis=1)
        self.multi_inout = np.concatenate((self.multi_inout, [[int(inner) * 2 - 1]]), axis=1)

    def fq(self, q):
        n_single, n_multi = self.single_qc.shape[1], self.multi_qc.shape[1]  # (1, n_single) and (7, n_multi)

        q = q.reshape(1, 7)
        q_scale_single = (q[0, self.single_mask.astype(np.int32)] * self.single_scale)  # (1, n_single)
        fq_single = (q_scale_single - self.single_qc) ** 2 - self.single_qbound ** 2  # (1, n_single)
        fqr_single = (q_scale_single - self.single_qc) ** 2 - self.single_qrbound ** 2  # (1, n_single)
        fq_single = fq_single * self.single_inout  # (1, n_single)
        fqr_single = fqr_single * self.single_inout  # (1, n_single)

        q = q.reshape(7, 1)
        q_scale_multi = q * self.multi_scale  # (7, n_multi)
        fq_multi = np.sum(((q_scale_multi - self.multi_qc) ** 2) * self.multi_mask, axis=0) - self.multi_qbound ** 2  # (1, n_multi)
        fqr_multi = np.sum(((q_scale_multi - self.multi_qc) ** 2) * self.multi_mask, axis=0) - self.multi_qrbound ** 2  # (1, n_multi)
        fq_multi = fq_multi * self.multi_inout  # (1, n_multi)
        fqr_multi = fqr_multi * self.multi_inout  # (1, n_multi)

        # print(fq_single, fqr_single, fq_multi, fqr_multi)

        return fq_single.reshape(1, n_single), fqr_single.reshape(1, n_single), \
                fq_multi.reshape(1, n_multi), fqr_multi.reshape(1, n_multi)

    def in_region(self, q):
        fq_single, _, fq_multi, _ = self.fq(q)

        return (fq_single <= 0), (fq_multi <= 0)

    def Ps(self, q):
        fq_single, fqr_single, fq_multi, fqr_multi = self.fq(q)
        Ps = 0
        Ps = Ps + np.sum(0.5 * self.single_kq * (np.minimum(0, fq_single)) ** 2)
        Ps = Ps + np.sum(0.5 * self.single_kr * (np.minimum(0, fqr_single)) ** 2)
        Ps = Ps + np.sum(0.5 * self.multi_kq * (np.minimum(0, fq_multi)) ** 2)
        Ps = Ps + np.sum(0.5 * self.multi_kr * (np.minimum(0, fqr_multi)) ** 2)

        return Ps

    def kesi_q(self, q):
        fq_single, fqr_single, fq_multi, fqr_multi = self.fq(q)

        partial_f_q = np.zeros((1, 7))
        q = q.reshape(1, 7)
        q_scale_single = (q[0, self.single_mask.astype(np.int32)] * self.single_scale)  # (1, n_single)
        for i, idx in enumerate(self.single_mask[0]):
            if fq_single[0, i] < 0:
                partial_f_q[0, idx] += 2 * (q_scale_single[0, i] - self.single_qc[0, i]) * self.single_inout[0, i]

        q = q.reshape(7, 1)
        q_scale_multi = q * self.multi_scale  # (7, n_multi)
        not_in_region_mask = (fq_multi < 0).repeat(7, 0)  # (7, n_multi)  [IMPORTANT] 
        partial_f_q += np.sum(2 * (q_scale_multi - self.multi_qc) * self.multi_inout * self.multi_mask * not_in_region_mask, axis=1).reshape(1, 7)

        kesi_q = np.zeros((1, 7))
        kesi_q = kesi_q + np.sum(self.single_kq * np.minimum(0, fq_single))
        kesi_q = kesi_q + np.sum(self.single_kr * np.minimum(0, fqr_single))
        kesi_q = kesi_q + np.sum(self.multi_kq * np.minimum(0, fq_multi))
        kesi_q = kesi_q + np.sum(self.multi_kr * np.minimum(0, fqr_multi))
        kesi_q = kesi_q * partial_f_q

        return kesi_q.reshape(1, -1)


class AdaptiveImageJacobian(object):
    """
        @ Class: AdaptiveImageJacobian
        @ Function: adaptive update the image jacobian
    """
    # def __init__(self, fa: FrankaArm =None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
    def __init__(self, fa=None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
        # n_k_per_dim => the number of rbfs in each dimension
        if fa is None:
            raise ValueError('FrankaArm handle is not provided!')
        self.fa = fa

        # dimensions declaration
        self.m = 6  # the dimension of Cartesian space configuration r
        self.n_k = n_k_per_dim ** 3  # the dimension of rbf function ??(r)
        # Js has three variables (x, y, z), 
        # and (??i, ??j, ??k) do not actually affect Js, 
        # when r is represented as body twist

        # Js here transforms Cartesian space BODY TWIST (v_b, w_b): 6 * 1 
        # to image space velocity (du, dv): 2 * 1
        # Js is 2 * 6, while Js[3:, :] = [0] because w_b do not actually affect (du, dv)
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
            if L.shape != (self.n_k, self.n_k):  # (1000, 1000)
                raise ValueError('Dimension of L should be ' + str((self.n_k, self.n_k)) + '!')
            self.L = L
        else:
            raise ValueError('Matrix L should not be empty!')

        if W_hat is not None:
            if W_hat.shape != (2 * self.m, self.n_k):  # (12, 1000)
                raise ValueError('Dimension of W_hat should be ' + str((2 * self.m, self.n_k)) + '!')
            self.W_hat = W_hat
        else:
            raise ValueError('Matrix W_hat should not be empty!')

        self.theta = RadialBF()
        self.theta.init_rbf_()

        self.image_space_region = ImageSpaceRegion()
        self.cartesian_space_region = CartesianSpaceRegion()
        self.cartesian_quat_space_region = CartesianQuatSpaceRegion()
        self.joint_space_region = JointSpaceRegion()

    def kesi_x(self, x):
        return self.image_space_region.kesi_x(x.reshape(1, -1))

    def kesi_r(self, r):
        return self.cartesian_space_region.kesi_r(r.reshape(1, -1))

    def kesi_rq(self, rq):
        return self.cartesian_quat_space_region.kesi_rq(Quat(rq.reshape(-1,)))

    def kesi_q(self, q):
        return self.joint_space_region.kesi_q(q.reshape(1, 7))

    def get_theta(self, r):
        return self.theta.get_rbf_(r)

    def get_Js_hat(self):
        return self.Js_hat

    def update(self):
        r = self.fa.get_pose()  # get r: including translation and rotation matrix
        q = self.fa.get_joints()  # get q: joint angles

        # split Cartesian translation and quaternion
        r_tran = r.translation
        r_quat = r.quaternion  # w, x, y, z

        theta = self.get_theta(r_tran)  # get the neuron values theta(r) (1000*1)
        J = self.fa.get_jacobian(q)  # get the analytic jacobian (6*7)
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        J_rot = Quat(r_quat).jacobian_rel2_axis_angle_()  # get the jacobian (partial p / partial r_o^T) (4, 3)
        
        kesi_x = self.kesi_x().reshape(-1, 1)  # (2, 1)
        kesi_rt = self.kesi_r().reshape(-1, 1)  # (3, 1)
        kesi_rq = self.kesi_rq() @ J_rot  # (1, 4) @ (4, 3) = (1, 3)
        kesi_r = np.r_[kesi_rt, kesi_rq.reshape(3, 1)]  # (6, 1)
        kesi_q = self.kesi_q().reshape(-1, 1)  # (7, 1)

        kesi_x_pie = np.c_[kesi_x[0] * np.eye(6), kesi_x[1] * np.eye(6)]  # (6, 12)

        dW_hat = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_r + J_pinv.T @ kesi_q).T  # (1000, 6)
        dW_hat = dW_hat @ kesi_x_pie  # (1000, 12)

        self.W_hat = self.W_hat + dW_hat.T

        # update J_s
        temp_Js_hat = self.W_hat @ theta  # (12, 1)
        self.Js_hat = np.c_[temp_Js_hat[:6], temp_Js_hat[6:]].T  # (2, 6)

        # deprecated
        '''
        dW_hat1_T = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_r + J_pinv.T @ kesi_q).T * kesi_x[0]
        dW_hat2_T = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_r + J_pinv.T @ kesi_q).T * kesi_x[1]

        dW_hat1, dW_hat2 = dW_hat1_T.T, dW_hat2_T.T  # (6, n_k), (6, n_k)
        dW_hat = np.concatenate((dW_hat1, dW_hat2), axis=0)  # (12, n_k)
        self.W_hat = self.W_hat + dW_hat  # (12, n_k)

        Js_hat1 = self.W_hat[:6, :] @ theta  # (6, 1)
        Js_hat2 = self.W_hat[6:, :] @ theta  # (6, 1)
        Js_hat = np.concatenate((Js_hat1.reshape(1, -1), Js_hat2.reshape(1, -1)), axis=0)  # (2, 6)
        self.Js_hat = Js_hat
        '''

class KnownImageJacobian(object):
    """
        @ Class: AdaptiveImageJacobian
        @ Function: adaptive update the image jacobian
    """
    # def __init__(self, fa: FrankaArm =None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
    def __init__(self, fa=None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
        # n_k_per_dim => the number of rbfs in each dimension
        if fa is None:
            raise ValueError('FrankaArm handle is not provided!')
        self.fa = fa

        # dimensions declaration
        self.m = 6  # the dimension of Cartesian space configuration r
        self.n_k = n_k_per_dim ** 3  # the dimension of rbf function ??(r)

        if Js is not None:
            if Js.shape != (2, 6):
                raise ValueError('Dimension of Js should be (2, 6), not ' + str(Js.shape) + '!')
            self.Js_hat = Js
        else:  # control with precise Js
            if x is None:
                # raise ValueError('Target point x on the image plane should not be empty!')
                x = np.array([1440/2,1080/2])
            fx, fy = MyConstants.FX_HAT, MyConstants.FY_HAT
            u0, v0 = MyConstants.U0, MyConstants.V0
            u, v = x[0] - u0, x[1] - v0
            z = 1
            self.J_cam2img = np.array([[fx/z, 0, -u/z, 0, 0, 0], \
                                  [0, fy/z, -v/z, 0, 0, 0]])
            self.R_c2b = np.array([[-1, 0,  0],
            [0, 1, 0],
            [0, 0, -1]])            
            
            self.p_s_in_panda_EE = np.array([0.067, 0.08, -0.05])
            ee_pose_quat = fa.get_pose().quaternion[[1,2,3,0]]
            ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
            p_s = ee_pose_mat @ self.p_s_in_panda_EE.reshape(3,1)
            p_s_cross = np.array([[0, -p_s[2,0], p_s[1,0]], \
                                [p_s[2,0], 0, -p_s[0,0]], \
                                [-p_s[1,0], p_s[0,0], 0]])
            J_p_cross = np.block([[np.eye(3),p_s_cross],[np.zeros((3,3)),np.zeros((3,3))]])

            self.Js_hat = self.J_cam2img @ np.block([[self.R_c2b,np.zeros((3,3))],[np.zeros((3,3)),self.R_c2b]]) @ J_p_cross # Js_hat = J_base2img
            
            # ========================
            Js_hat_for_init = np.array([[-1000,0,-1000,-1000,-1000,1000],[0,1000,1000,-1000,-1000,1000]])
            self.Js_hat = Js_hat_for_init
        # now we do not update Js!! yxj 0630
        # if L is not None:
        #     if L.shape != (self.n_k, self.n_k):  # (1000, 1000)
        #         raise ValueError('Dimension of L should be ' + str((self.n_k, self.n_k)) + '!')
        #     self.L = L
        # else:
        #     raise ValueError('Matrix L should not be empty!')

        # if W_hat is not None:
        #     if W_hat.shape != (2 * self.m, self.n_k):  # (12, 1000)
        #         raise ValueError('Dimension of W_hat should be ' + str((2 * self.m, self.n_k)) + '!')
        #     self.W_hat = W_hat
        # else:
        #     raise ValueError('Matrix W_hat should not be empty!')
        cfg = {'n_dim':3,'n_k_per_dim':10,'sigma':1,'pos_restriction':np.array([[-0.1,0.9],[-0.5,0.5],[0,1]])}
        self.theta = RadialBF(cfg=cfg)
        self.theta.init_rbf_()

        self.image_space_region = ImageSpaceRegion(b=np.array([1440,1080]))
        self.cartesian_space_region = CartesianSpaceRegion()
        self.cartesian_quat_space_region = CartesianQuatSpaceRegion()
        self.joint_space_region = JointSpaceRegion()

        self.cartesian_space_region.set_r_c(MyConstants.CARTESIAN_CENTER)
        self.cartesian_space_region.set_c(np.array([0.02, 0.02, 0.02]).reshape(1, 3))
        self.cartesian_space_region.set_Kc(np.array([0,0,0]).reshape(1, 3))

        self.cartesian_quat_space_region.set_q_g(np.array([-0.2805967680249283, 0.6330528569977758, 0.6632800072901188, 0.2838309407825178]))  # grasping pose on the right
        self.cartesian_quat_space_region.set_Ko(0)

    def kesi_x(self, x):
        return self.image_space_region.kesi_x(x.reshape(1, -1))

    def kesi_r(self, r):
        return self.cartesian_space_region.kesi_r(r.reshape(1, -1))

    def kesi_rq(self, rq):
        return self.cartesian_quat_space_region.kesi_rq(Quat(rq.reshape(-1,)))

    def kesi_q(self, q):
        return self.joint_space_region.kesi_q(q.reshape(1, 7))

    def get_theta(self, r):
        return self.theta.get_rbf_(r)

    def get_Js_hat(self, x, p_s=None):
        # x is the pixel coordinate on the image plane
        # p_s is the vector pointing from {NE}_origin to ArUco marker center
        return self.Js_hat

    def get_u(self,J,d,r_t,r_o,q,x,with_vision=False, p_s=None):
        J_pinv = J.T @ np.linalg.pinv(J @ J.T)

        kesi_x = self.kesi_x(x).reshape(-1, 1)  # (2, 1)

        kesi_r = self.kesi_r(r_t.reshape(1, 3))  # (1, 3)

        fx, fy = MyConstants.FX_HAT, MyConstants.FY_HAT
        u0, v0 = MyConstants.U0, MyConstants.V0
        u, v = x[0] - u0, x[1] - v0
        z = 1
        self.J_cam2img = np.array([[fx/z, 0, -u/z, 0, 0, 0], \
                                [0, fy/z, -v/z, 0, 0, 0]])
        ee_pose_quat = fa.get_pose().quaternion[[1,2,3,0]]
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        p_s = ee_pose_mat @ self.p_s_in_panda_EE.reshape(3,1)
        p_s_cross = np.array([[0, p_s[2,0], -p_s[1,0]], \
                            [-p_s[2,0], 0, p_s[0,0]], \
                            [p_s[1,0], -p_s[0,0], 0]])
        # J_p_cross = np.block([[np.eye(3),p_s_cross],[np.zeros((3,3)),np.zeros((3,3))]])
        Jvel = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), p_s_cross]])

        self.Js_hat = self.J_cam2img @ np.block([[self.R_c2b, self.R_c2b], [np.zeros((3, 6))]]) @ Jvel

        # print("self.J_cam2img",self.J_cam2img)
        # print("2",np.block([[self.R_c2b,np.zeros((3,3))],[np.zeros((3,3)),self.R_c2b]]))
        # print("3",J_p_cross)
        # print("self.Js_hat",self.Js_hat)

        if with_vision:
            kesi_rq = np.zeros((1, 3))
        else:
            if self.cartesian_quat_space_region.fo(Quat(r_o)) <= 0:
                kesi_rq = np.zeros((1, 3))
            else:
                kesi_rq = self.cartesian_quat_space_region.kesi_rq_omega(r_o) / 2 # (1, 3)
        kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)
        self.kesi_rall = kesi_rall

        kesi_q = self.kesi_q(q).reshape(7, 1)  # (7, 1)

        if with_vision:
            Js_hat = self.get_Js_hat(x, p_s=p_s)
            
            u = - J_pinv @ (Js_hat.T @ kesi_x + kesi_rall + J @ kesi_q)  # normal version in paper
            # Js_hat_pinv = Js_hat.T @ np.linalg.inv(Js_hat @ Js_hat.T)  # try pseudo inverse of Js
            # u = - J_pinv @ (Js_hat_pinv @ kesi_x + kesi_rall + J_pinv.T @ kesi_q)
        else:
            u = - J_pinv @ (kesi_rall + J @ kesi_q)
        return u

def test_vision_region_control(fa):
    pre_traj = "./data/0708/my_vision_region_"+time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))+"/"
    os.mkdir(pre_traj)
    class vision_collection(object):
        def __init__(self) -> None:
            self.vision_1_ready = False
            self.vision_2_ready = False
            self.x1 = np.array([-1000,-1000])
            self.x2 = np.zeros((2,))

        def vision_1_callback(self, msg):
            self.x1 = np.array([msg.point.x,msg.point.y])
            if not self.vision_1_ready:
                self.vision_1_ready = True

        def vision_2_callback(self, msg):
            self.x2 = np.array([msg.point.x,msg.point.y])
            if not self.vision_2_ready:
                self.vision_2_ready = True

        def is_data_without_vision_1_ready(self):
            return self.vision_2_ready

        def is_data_with_vision_1_ready(self):
            return self.vision_1_ready & self.vision_2_ready
    
    desired_position_bias = np.array([-200, -100])
    data_c = vision_collection()
    # controller_x = ImageSpaceRegion(b=np.array([1920,1080]))
    controller_precise = KnownImageJacobian(fa)

    sub_vision_1_ = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, data_c.vision_1_callback, queue_size=1)
    sub_vision_2_ = rospy.Subscriber('/aruco_simple/pixel2', PointStamped, data_c.vision_2_callback, queue_size=1)
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate_ = rospy.Rate(100)

    # move to the place where the camera can see the ee aruco marker
    target_joint = [0.80492297,-0.58386596,0.30586097,-2.71391148,0.64828513,2.52442803,-0.0742324]# 0704 middle->center
    # target_joint = [1.47167024, -0.60982986,  0.29298495, -2.49924619,  0.10524934,  2.19492362,  1.01318275] # 0708 right->center
    # target_joint = [1.25479051, -0.20090372, -0.24059873, -2.47008572,  0.00602366,  2.44270354,  0.54778121] # 0708 left->center
    fa.goto_joints(target_joint, joint_impedances=FC.DEFAULT_JOINT_IMPEDANCES,ignore_virtual_walls = True)

    # init control scheme
    print('i have reached the desired initial pos!')
    if data_c.is_data_with_vision_1_ready():
        target = data_c.x2
        target[0] = target[0]+desired_position_bias[0]
        target[1] = target[1]+desired_position_bias[1]
        controller_precise.image_space_region.set_x_d(target)
        controller_precise.image_space_region.set_Kv(0.2)
        print('Vision region is set!')


    f_list, p_list, kesi_x_list, pixel_1_list, pixel_2_list, time_list=[], [], [], [], [], []
    q_and_manipubility_list = np.zeros((0, 8))

    fx = MyConstants.FX_HAT
    fy = MyConstants.FY_HAT
    u0 = MyConstants.U0
    v0 = MyConstants.V0  
    
    R_c2b =  np.array([[-1, 0,  0],
            [0, 1, 0],
            [0, 0, -1]])      
    depth = 1


    # print('Js',Js)
    max_execution_time = 20.0

    home_joints = fa.get_joints()
    fa.dynamic_joint_velocity(joints=home_joints,
                                joints_vel=np.zeros((7,)),
                                duration=max_execution_time,
                                buffer_time=10,
                                block=False)
    i=0

    start_time = time.time() 
    # update control scheme
    while not rospy.is_shutdown():
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = fa.get_joints()
        J = fa.get_jacobian(q_and_m[0, :7])
        det = np.linalg.det(J @ J.T)
        q_and_m[0, 7] = math.sqrt(np.abs(det))
        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)
        pose = fa.get_pose()
        
        kesi_x = controller_precise.image_space_region.kesi_x(data_c.x1)
        kesi_x = kesi_x.reshape((2,1))

        d = np.array([[0],[0],[0],[0],[0],[0]])

        dq_d_ = controller_precise.get_u(J,d,pose.translation,pose.quaternion,q_and_m[0, :7],data_c.x1, with_vision=True)

        time_now = rospy.Time.now().to_time() - start_time
        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=i, timestamp=time_now, 
            seg_run_time=max_execution_time,
            joints=home_joints,
            joint_vels=dq_d_.reshape(7,).tolist()
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        i += 1
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)

        # logging
        time_list.append(time.time()-start_time)
        pixel_1_list.append(data_c.x1)
        pixel_2_list.append(data_c.x2)
        f_list.append(controller_precise.image_space_region.fv(data_c.x1))
        p_list.append(controller_precise.image_space_region.Pv(data_c.x1))
        kesi_x_list.append(kesi_x.reshape(2,))

        if time.time() - start_time >= max_execution_time:
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - start_time, should_terminate=True)
            ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
            pub.publish(ros_msg)
            break

            # print(kesi_r)
            # print(kesi_rq)
        
        rate_.sleep()
    
    # vision part
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time_list, f_list, color='b',label = 'f')
    plt.plot(time_list, p_list, color='r',label = 'P')
    plt.title('f and P for vision region')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(time_list,kesi_x_list,label = 'kesi')
    plt.title('kesi_x')
    plt.savefig(pre_traj+'f_P_kesi_x.jpg')

    plt.figure()
    plt.plot(time_list, pixel_1_list,color='b',label = 'vision position')
    plt.plot(time_list, pixel_2_list+desired_position_bias,color='r',label = 'desired position')
    plt.legend()
    plt.ylim([0,1440])
    plt.title('vision position vs time')
    plt.savefig(pre_traj+'vision_position.jpg')

    plt.figure()
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='b',label = 'vision trajectory')
    plt.scatter(target[0], target[1],color='r',label = 'desired position')
    plt.xlim([0,1440])
    plt.ylim([0,1080])
    ax = plt.gca()
    ax.invert_yaxis()
    plt.legend()
    plt.title('vision trajectory')
    plt.savefig(pre_traj+'vision_trajectory.jpg')

    plt.show()
    info = {'f_list': f_list, \
            'p_list': p_list, \
            'kesi_x_list': kesi_x_list, \
            'pixel_1_list': pixel_1_list, \
            'pixel_2_list': pixel_2_list, \
            'time_list': time_list, \
            'q_and_manipubility_list':q_and_manipubility_list}
    with open(pre_traj + 'data.pkl', 'wb') as f:
        pickle.dump(info, f)


if __name__ == '__main__':
    # jsr = JointSpaceRegion()
    # jsr.add_region_multi(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 0.08, 0.1, np.array([1, 1, 1, 1, 1, 1, 1]), kq=5000, kr=10, inner=True)
    
    # q_test = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # fq_s, fqr_s, fq_m, fqr_m = jsr.fq(q_test)
    # print(fq_s, fqr_s, fq_m, fqr_m)
    # fq_sin, fqr_sin = jsr.in_region(q_test)
    # print(fq_sin, fqr_sin)
    # Ps = jsr.Ps(q_test)
    # print(Ps)
    # kesi_q = jsr.kesi_q(q_test)
    # print(kesi_q)

    # q_test_list = []
    # qd_list = []
    # ps_list = []
    # for qd in np.linspace(0.5, 0.75, 400):
    #     q_test = np.array([0.5, 0.5, qd, qd, 0.5, 0.5, 0.5])
    #     ps = jsr.Ps(q_test)
    #     kesi_q = jsr.kesi_q(q_test)
    #     q_test_list.append(kesi_q.squeeze())
    #     qd_list.append(qd)
    #     ps_list.append(ps)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(qd_list, ps_list)
    # plt.ylim(0, 0.005)
    # plt.show()

    fa = FrankaArm()
    test_vision_region_control(fa=fa)
    # plot_figures()
    
