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
            r∈[-pi, pi], p∈[-pi/2, pi/2], y∈[-pi, pi], which is euler angle in the order of 'XYZ'
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
        self.n_k = n_k_per_dim ** 3  # the dimension of rbf function θ(r)
        # Js has three variables (x, y, z), 
        # and (θi, θj, θk) do not actually affect Js, 
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

class JointOutputRegionControl(object):
    def __init__(self, sim_or_real='sim', fa=None) -> None:
        self.joint_space_region = JointSpaceRegion()
        self.sim_or_real = sim_or_real
        if sim_or_real == 'real':
            if fa is None:
                raise ValueError('FrankaArm handle is not provided!')
            self.fa = fa
        self.cd = np.eye(7)

        self.joint_space_region = JointSpaceRegion()
        self.init_joint_region()

    def init_joint_region(self):
        # self.joint_space_region.add_region_multi(qc=np.array([0.7710433735413842, -0.30046383584419534, 0.581439764761865, -2.215658436023894, 0.4053359101811589, 2.7886962782933757, 0.4995608692842497]), \
        #                                          qbound=0.12, qrbound=0.10, \
        #                                          mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
        #                                          kq=10, kr=0.1, \
        #                                          inner=False, scale=np.ones((7, 1)))

        # self.joint_space_region.add_region_multi(qc=np.array([0.35612043, -0.42095495, 0.31884579, -2.16397413, 0.36776436,2.10526431, 0.45843472]), \
        #                                          qbound=0.08, qrbound=0.10, \
        #                                          mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
        #                                          kq=100000, kr=1000, \
        #                                          inner=True, scale=np.ones((7, 1)))

        self.joint_space_region.add_region_multi(qc=np.array([1.19876445, 0.16743403, 0.46827566, -2.40414747, 0.47512512, 3.3847505, 0.42326836]), \
                                                 qbound=0.12, qrbound=0.10, \
                                                 mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
                                                 kq=1, kr=0.01, \
                                                 inner=False, scale=np.ones((7, 1)))
        self.singularity_joint = np.array([1.19582476e+00, -1.79016522e-03, 3.56311106e-01, -2.51608346e+00, 4.75119009e-01, 3.31746127e+00, 5.93365287e-01])
        self.joint_space_region.add_region_multi(qc=self.singularity_joint, \
                                                 qbound=0.50, qrbound=0.45, \
                                                 mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
                                                 kq=1000000, kr=10000, \
                                                 inner=True, scale=np.ones((7, 1)))# this is joint sigularity position: inner = True
    def calc_manipubility(self, J_b):
        det = np.linalg.det(J_b @ J_b.T)
        return math.sqrt(np.abs(det))

    def get_dq_d_(self, q:np.ndarray, d:np.ndarray=np.zeros((7, 1)), J_sim:np.ndarray=None, time_start_this_loop=None):
        q, d = q.reshape(7,), d.reshape(7, 1)
        if self.sim_or_real == 'real':
            J = self.fa.get_jacobian(q) # get the analytic jacobian (6*7)
            # print(J) 
        elif self.sim_or_real == 'sim':
            J = J_sim
        # print('time consumption2: ', time.time() - time_start_this_loop)
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        # print('time consumption3: ', time.time() - time_start_this_loop)
        N = np.eye(7) - J_pinv @ J  # get the zero space matrix (7*7)
        kesi_q = self.joint_space_region.kesi_q(q).reshape(7, 1)

        dq_d = - J_pinv @ (J @ kesi_q) + N @ np.linalg.inv(self.cd) @ d  # (7, 1)
        # dq_d = - kesi_q

        return dq_d, kesi_q

def test_cartesian_joint_space_region_control(fa):
    controller_j = JointSpaceRegion()
    controller_r = CartesianSpaceRegion()
    controller_rq = CartesianQuatSpaceRegion()

    # init control scheme
    # controller_r.set_r_c(np.array([-0.4274491954570557, 0.17649338322602143, 0.0877387993047109]).reshape(1, 3))
    # controller_r.set_r_c(np.array([0.30705422269100857, -7.524700079232461e-06, 0.4870834722065375]))  # starting point
    # controller_r.set_r_c(np.array([-0.01624475011413961, 0.5786721263542499, 0.30532807964440667]))  # grasping pose on the right
    controller_r.set_r_c(np.array([0.03493165, 0.7115742, 0.27140488]))  # grasping pose real robot on the right
    controller_r.set_c(np.array([0.01, 0.01, 0.01]).reshape(1, 3))
    controller_r.set_Kc(np.array([1e-7, 1e-7, 1e-7]).reshape(1, 3))

    # controller_rq.set_q_g(np.array([0.018123963264730075, 0.9941285663016644, -0.0020305968914211114, 0.10663487821459938]))
    # controller_rq.set_q_g(np.array([0.0004200682001275777, 0.9028032461428314, -0.00021468888469159875, -0.4300479986956025]))  # rot_Y(-90)
    # controller_rq.set_q_g(np.array([0.0005740008069370812, 0.708744905485196, -0.7054320813712253, 0.00603014709058085]))  # rot_Z(+90)
    # controller_rq.set_q_g(np.array([0.5396668336145884, -0.841829320383993, 0.00250734600652218, -0.008486521399793301]))  # rot_X(+90)
    # controller_rq.set_q_g(np.array([-0.2805967680249283, 0.6330528569977758, 0.6632800072901188, 0.2838309407825178]))  # grasping pose on the right
    
    controller_rq.set_q_g(np.array([-0.32260996,  0.67001947,  0.59924927,  0.29645973]))  # grasping pose real robot on the right
    controller_rq.set_Ko(15)
    
    pre_traj = "./data/0610/cartesian_region/"
    singularity_joint = np.array([0.897782, 0.20624612, 0.53392278, -2.34249171, -0.20088835, 3.2333697, 0.97114771])
    # have singularity
    controller_j.add_region_multi(qc=singularity_joint, \
                                qbound=0.2, qrbound=0.70, \
                                mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
                                kq=400, kr=10, \
                                inner=True, scale=np.ones((7, 1)))# this is joint sigularity position: inner = True
    pre_traj = "./data/0610/cartesian_region_have_joint_sing_2/"

    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate_ = rospy.Rate(100)

    f_quat_list, p_quat_list, quat_list = [], [], []
    f_pos_list, p_pos_list, pos_list = [], [], []
    dist_list = []
    q_and_manipubility_list = np.zeros((0, 8))
    t_list = []

    max_execution_time = 20.0

    home_joints = fa.get_joints()
    fa.dynamic_joint_velocity(joints=home_joints,
                                joints_vel=np.zeros((7,)),
                                duration=max_execution_time,
                                buffer_time=10,
                                block=False)

    i=0

    time_start = rospy.Time.now().to_time()

    # update control scheme
    while not rospy.is_shutdown():
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = fa.get_joints()
        J = fa.get_jacobian(q_and_m[0, :7])
        det = np.linalg.det(J @ J.T)
        q_and_m[0, 7] = math.sqrt(np.abs(det))
        pose = fa.get_pose()
        
        dist = np.linalg.norm((singularity_joint - q_and_m[0, :7]), ord=2)
        dist_list.append(dist)
        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)

        kesi_r = controller_r.kesi_r(pose.position.reshape(1, 3))  # (1, 3)
        # kesi_rq = controller_rq.kesi_rq(pose.quaternion) / 50  # (1, 3)
        if controller_rq.fo(Quat(pose.quaternion)) <= 0:
            kesi_rq = np.zeros((1, 3))
        else:
            kesi_rq = controller_rq.kesi_rq_omega(pose.quaternion) / 2 # (1, 3)

        kesi_q = controller_j.kesi_q(q_and_m[0, :7]).reshape(7, 1)  # (7, 1)
        # print(kesi_q)
        kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)

        # convertion between spatial Twist and body Twist
        rotMat = pose.rotation
        p = pose.position
        pMat = np.array([[0, -p[2], p[1]], \
                            [p[2], 0, -p[0]], \
                            [-p[1], p[0], 0]])
        T = np.block([[rotMat, np.zeros((3, 3))], [pMat @ rotMat, rotMat]])

        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        
        dq_d_ = -J_pinv @ (kesi_rall + J @ kesi_q)


        # msg = Float64MultiArray()
        # msg.data = dq_d_.reshape(7,)
        # pub_.publish(msg)
        time_now = rospy.Time.now().to_time() - time_start
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

        f_quat_list.append(controller_rq.fo(Quat(pose.quaternion)))
        p_quat_list.append(controller_rq.Po(Quat(pose.quaternion)))
        f_pos_list.append(controller_r.fc(p).reshape(-1,))
        p_pos_list.append(controller_r.Pc(p).reshape(-1,))
        quat_list.append(pose.quaternion.tolist())
        t_list.append(time_now)
        pos_list.append(p)
        
        if time.time() - time_start >= max_execution_time:
            # terminate dynamic joint velocity control
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - time_start, should_terminate=True)
            ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
            pub.publish(ros_msg)
            break

        print(kesi_r)
        print(kesi_rq)
        
        rate_.sleep()

    with open('/home/roboticslab/yxj/frankapy/data/0610/cartesian_region/all_data.pkl', 'rb') as f1:
        data2 = pickle.load(f1)
        q_m_comp = data2['q_and_manipubility_list']
        t_list_comp = data2['t_list']

    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.plot(t_list,f_quat_list, color='b',label="fo")
    plt.plot(t_list,p_quat_list, color='r',label="Po")
    ax.set_title("f and P for quat")
    ax.legend()
    ax = plt.subplot(2, 2, 2)
    plt.plot(t_list,quat_list)
    ax.set_title("quat")
    ax.legend(['w','x','y','z'])
    ax = plt.subplot(2, 2, 3)
    plt.plot(t_list,f_pos_list, color='b',label="fc")
    plt.plot(t_list,p_pos_list, color='r',label="Pc")
    ax.set_title("f and P for pos")
    ax.legend(['x','y','z'])
    ax = plt.subplot(2, 2, 4)
    plt.plot(t_list, pos_list)
    ax.set_title("pos")
    ax.legend(['x','y','z'])

    plt.savefig(pre_traj+"pos_quat.jpg")

    plt.figure()
    for i in range(8):
        ax = plt.subplot(4, 2, i+1)
        plt.plot(t_list, q_and_manipubility_list[:,i])
        if i<7:
            plt.plot(t_list_comp, q_m_comp[:,i])
            ax.legend(['joint %d'%(i+1),'comp'])
        else:
            ax.legend(['manipubility'])
            plt.plot(t_list_comp, q_m_comp[:,i])
            ax.legend(['manipubility','manipubility_comp'])
    plt.savefig(pre_traj+'q_manip.jpg')

    plt.show()

    info = {'dist': dist_list, \
            'q_and_manipubility_list': q_and_manipubility_list, \
            'p_quat_list': p_quat_list, \
            'f_quat_list': f_quat_list, \
            'quat_list': quat_list, \
            'p_pos_list': p_pos_list, \
            'f_pos_list': f_pos_list, \
            'pos_list': pos_list, \
            't_list': t_list}
    with open(pre_traj+'all_data.pkl', 'wb') as f:
        pickle.dump(info, f)

# 0630 yxj
def test_adaptive_region_control(fa):
    class data_collection(object):
        def __init__(self) -> None:
            self.J = np.zeros((6, 7))
            self.q = np.zeros((7,))
            self.trans = np.zeros((3,))
            self.quat = np.zeros((4,))
            self.J_ready = False
            self.q_ready = False
            self.pose_ready = False
            self.vision_1_ready = False
            self.vision_2_ready = False
            self.x1 = np.array([-1000,-1000])
            self.x2 = np.zeros((2,))

        def zero_jacobian_callback(self, msg):
            self.J = np.array(msg.data).reshape(6, 7)
            if not self.J_ready:
                self.J_ready = True

        def joint_angles_callback(self, msg):
            self.q = np.array(msg.data).reshape(7,)
            if not self.q_ready:
                self.q_ready = True

        def pose_callback(self, msg):
            self.trans = np.array(msg.data)[:3].reshape(3,)
            self.quat = np.array(msg.data)[[6, 3, 4, 5]].reshape(4,)
            if not self.pose_ready:
                self.pose_ready = True

        def vision_1_callback(self, msg):
            self.x1 = np.array([msg.point.x,msg.point.y])
            if not self.vision_1_ready:
                self.vision_1_ready = True

        def vision_2_callback(self, msg):
            self.x2 = np.array([msg.point.x,msg.point.y])
            if not self.vision_2_ready:
                self.vision_2_ready = True

        def is_data_without_vision_1_ready(self):
            return self.J_ready & self.q_ready & self.pose_ready & self.vision_2_ready

        def is_data_with_vision_ready(self):
            return self.J_ready & self.q_ready & self.pose_ready & self.vision_1_ready & self.vision_2_ready

    data_c = data_collection()
    controller_adaptive = AdaptiveImageJacobian(fa)
    
    nh_ = rospy.init_node('cartesian_joint_space_region_testbench', anonymous=True)
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    sub_q_ = rospy.Subscriber('/gazebo_sim/joint_angles', Float64MultiArray, data_c.joint_angles_callback, queue_size=1)
    sub_ee_pose_ = rospy.Subscriber('/gazebo_sim/ee_pose', Float64MultiArray, data_c.pose_callback, queue_size=1)
    sub_vision_1_ = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, data_c.vision_1_callback, queue_size=1)
    sub_vision_2_ = rospy.Subscriber('/aruco_simple/pixel2', PointStamped, data_c.vision_2_callback, queue_size=1)
    rate_ = rospy.Rate(100)

    # init control scheme
    while 1:
        if data_c.is_data_without_vision_1_ready():
            controller_adaptive.image_space_region.set_x_d(data_c.x2)
            # print("1111",controller_adaptive.image_space_region.x_d)
            # controller_adaptive.image_space_region.set_b(50)
            controller_adaptive.image_space_region.set_Kv(0.2)
            print('vision region is set!')
            break


    f_list, p_list, kesi_x_list, pixel_1_list, pixel_2_list, time_list=[], [], [], [], [], []
    q_and_manipubility_list = np.zeros((0, 8))
    f_quat_list,p_quat_list,quat_list,kesi_rall_list,position_list = [],[],[],[],[]

    start_time = time.time() 
    
    # update control scheme
    while not rospy.is_shutdown():
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = data_c.q
        q_and_m[0, 7] = math.sqrt(np.abs(np.linalg.det(data_c.J @ data_c.J.T)))

        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)

        J = data_c.J
        # J_pos = J[:3,:] # 3x7
        # J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7x6)
        # J_pos_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T) # 7x3
        
        # u = data_c.x1[0]-u0
        # v = data_c.x1[1]-v0
        # Js = np.array([[fx/depth,0,u/depth],[0,fy/depth,v/depth]]) @ R_c2b # 2x3

        d = np.array([[0],[0],[0],[0],[0],[0]])

        dq_d_ = controller_adaptive.get_u(J,d,data_c.trans,data_c.quat,data_c.q,data_c.x1)
        msg = Float64MultiArray()
        msg.data = dq_d_.reshape(7,)
        pub_.publish(msg)

        # logging
        time_list.append(time.time()-start_time)
        pixel_1_list.append(data_c.x1)
        pixel_2_list.append(data_c.x2)
        f_list.append(controller_adaptive.image_space_region.fv(data_c.x1))
        p_list.append(controller_adaptive.image_space_region.Pv(data_c.x1))
        kesi_x_list.append(controller_adaptive.image_space_region.kesi_x(data_c.x1).reshape((-1,)))

        f_quat_list.append(controller_adaptive.cartesian_quat_space_region.fo(Quat(data_c.quat)))
        p_quat_list.append(controller_adaptive.cartesian_quat_space_region.Po(Quat(data_c.quat)))
        quat_list.append(data_c.quat.tolist())
        kesi_rall_list.append(controller_adaptive.kesi_rall)
        position_list.append(data_c.trans)

        if time.time() - start_time >= 30.0:
            break

            # print(kesi_r)
            # print(kesi_rq)
        
        rate_.sleep()

    pre_traj = './data/0623/'
    
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
    plt.plot(time_list, pixel_2_list,color='r',label = 'desired position')
    plt.legend()
    plt.title('vision position vs time')
    plt.savefig(pre_traj+'vision_position.jpg')

    plt.figure()
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='b',label = 'vision trajectory')
    plt.scatter(pixel_2_list[0][0], pixel_2_list[0][1],color='r',label = 'desired position')
    plt.legend()
    plt.title('vision trajectory')
    plt.savefig(pre_traj+'vision_trajectory.jpg')

    # cartesian part
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time_list, f_quat_list, color='b',label = 'f_quat')
    plt.plot(time_list, p_quat_list, color='r',label = 'p_quat')
    plt.title('f and p for quaternion')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(time_list, quat_list)
    plt.title('quaternion vs time')
    plt.savefig(pre_traj+'cartesian_quat.jpg')

    plt.figure()
    ax1 = plt.axes(projection='3d')
    position_array  = np.array(position_list)
    ax1.plot3D(position_array[:,0],position_array[:,1],position_array[:,2],label='traj')
    ax1.scatter(position_array[0,0],position_array[0,1],position_array[0,2],c='r',label='initial')
    ax1.scatter(position_array[200,0],position_array[200,1],position_array[200,2],c='b',label='t=5s')
    ax1.scatter(-0.0068108842682527, 0.611158320250102, 0.5342875493162069,c='g',label='goal region center')
    ax1.legend()
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('executed trajectory 3D')
    plt.savefig(pre_traj+'cartesian_3d.jpg')

    plt.figure()
    plt.plot(time_list,np.reshape(kesi_rall_list,(np.shape(time_list)[0],-1)),label = 'kesi')
    plt.legend()
    plt.title('kesi for 6 dimensions')
    plt.savefig(pre_traj+'cartesian_kesi.jpg')

    plt.show()
    info = {'f_list': f_list, \
            'p_list': p_list, \
            'kesi_x_list': kesi_x_list, \
            'pixel_1_list': pixel_1_list, \
            'pixel_2_list': pixel_2_list, \
            'time_list': time_list, \
            'q_and_manipubility_list':q_and_manipubility_list,\
            'quat_list': quat_list,\
            'f_quat_list':f_quat_list,\
            'p_quat_list':p_quat_list,\
            'kesi_rall_list':kesi_rall_list,\
            'position_list':position_list}
    with open('./data/0623/data.pkl', 'wb') as f:
        pickle.dump(info, f)


def plot_figures():
    import pickle
    with open('./data/0521/q_and_manip.pkl', 'rb') as f:
        data1 = pickle.load(f)
    with open('./data/0521/q_and_manip.pkl', 'rb') as f:
        data2 = pickle.load(f)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(data1['q_and_manip'][:, :7], color='r')
    plt.plot(data2['q_and_manip'][:, :7], color='b')
    plt.title('distance to singularity')
    plt.legend(['with jsr', 'with no jsr'])
    plt.subplot(1, 2, 2)
    plt.plot(data1['q_and_manip'][:, 7], color='r')
    plt.plot(data2['q_and_manip'][:, 7], color='b')
    plt.title('manipubility')
    plt.legend(['with jsr', 'with no jsr'])
    plt.savefig('./data/0521/test_Cartesian_space_region.png', dpi=600)
    plt.show()


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
    # test_joint_space_region_control(fa=fa)
    test_cartesian_joint_space_region_control(fa=fa)
    # plot_figures()
    
