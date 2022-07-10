"""
    my_adaptive_control.py, 2022-05-01
    Copyright 2022 IRM Lab. All rights reserved.
"""
import collections
import logging
from matplotlib.contour import ContourLabeler
# from frankapy.franka_arm import FrankaArm
import rospy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import time
import pdb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
print(sys.path)
from my_utils import Quat, RadialBF
from matplotlib import pyplot as plt
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped
from matplotlib import pyplot as plt
import pickle


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

class MyConstantsSim(object):
    """
        @ Class: MyConstants
        @ Function: get all the constants in this file
    """
    FX_HAT = 3759.66467 + 800
    FY_HAT = 3759.66467 + 800
    U0 = 960.5 + 100
    V0 = 540.5 - 100

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
        # print(self.fv(x))
        partial_fv = 2 * (x - self.x_d) / (self.b ** 2) if self.fv(x)<=0 else np.array([[0],[0]])
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

    # deprecated (only valid when the last 3 dimensions in x_dot are represented in axis-angle form)
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
            # raise ValueError('FrankaArm handle is not provided!')
            print('-------simulation-------')
        else:
            self.fa = fa

        # dimensions declaration
        self.m = 6  # the dimension of Cartesian space configuration r
        self.n_k = n_k_per_dim ** 3  # the dimension of rbf function θ(r)
        # Js has three variables (x, y, z), 
        # and (θi, θj, θk) do not actually affect Js, 
        # when r is represented as body twist

        # Js here transforms Cartesian space BODY TWIST (v_b, w_b): 6 * 1 
        # to image space velocity (du, dv): 2 * 1
        # Js is 2 * 6, while Js[:,3:] = [0] because w_b do not actually affect (du, dv)
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
            my_jacobian_handler = ()
            J_base2cam = my_jacobian_handler.calcJacobian(from_frame='panda_link0', to_frame='camera_link')
            
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

        cfg = {'n_dim':3,'n_k_per_dim':10,'sigma':1,'pos_restriction':np.array([[-0.1,0.9],[-0.5,0.5],[0,1]])}
        self.theta = RadialBF(cfg=cfg)
        self.theta.init_rbf_()

        self.image_space_region = ImageSpaceRegion(b=np.array([1920,1080]))
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

        kesi_x_prime = np.c_[kesi_x[0] * np.eye(6), kesi_x[1] * np.eye(6)]  # (6, 12)

        dW_hat = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_r + J_pinv.T @ kesi_q).T  # (1000, 6)
        dW_hat = dW_hat @ kesi_x_prime  # (1000, 12)

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

# yxj 20220623
class AdaptiveRegionControllerSim(object):
    """
        @ Class: AdaptiveRegionControllerSim
        @ Function: copied and modified from AdaptiveImageJacobian, region controller with adaptive or precise Js
    """
    # def __init__(self, fa: FrankaArm =None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
    def __init__(self, fa=None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None, allow_adaptive=False) -> None:
        # n_k_per_dim => the number of rbfs in each dimension

        if fa is None:
            # raise ValueError('FrankaArm handle is not provided!')
            print('-------simulation-------')
            self.fa = None
        else:
            self.fa = fa

        # dimensions declaration
        self.m = 6  # the dimension of Cartesian space configuration r
        self.n_k = n_k_per_dim ** 3  # the dimension of rbf function θ(r)
        # Js has three variables (x, y, z), 
        # and (θi, θj, θk) do not actually affect Js, 
        # when r is represented as body twist

        # Js here transforms Cartesian space BODY TWIST (v_b, w_b): 6 * 1 
        # to image space velocity (du, dv): 2 * 1
        # Js is 2 * 6, while Js[:,3:] = [0] because w_b do not actually affect (du, dv)

        # calculation for image jacobian
        self.allow_adaptive = allow_adaptive
        self.R_b2c = np.array([[-1, 0,  0],
                               [0,  1,  0],
                               [0,  0, -1]])
        self.base_Js = np.array([[MyConstantsSim.FX_HAT/1, 0,    -MyConstantsSim.U0/1, 0, 0, 0], \
                                 [0,    MyConstantsSim.FY_HAT/1, -MyConstantsSim.V0/1, 0, 0, 0]])

        if Js is not None:
            if Js.shape != (2, 6):
                raise ValueError('Dimension of Js should be (2, 6), not ' + str(Js.shape) + '!')
            self.Js_hat = Js
        else:  # control with precise Js
            if x is None:
                # raise ValueError('Target point x on the image plane should not be empty!')
                x = np.array([1920/2,1080/2])

            # old version
            """
            # J_cam2img = np.array([[fx/z, 0,    -u/z, -u*v/fx,      (fx+u**2)/fx, -v], \
            #                       [0,    fy/z, -v/z, -(fy+v**2)/fy, u*v/fy,       u]])
            fx, fy = MyConstantsSim.FX_HAT, MyConstantsSim.FY_HAT
            u0, v0 = MyConstantsSim.U0, MyConstantsSim.V0
            z = 1
            J_cam2img = np.array([[fx/z, 0,    -u0/z, 0, 0, 0], \
                                  [0,    fy/z, -v0/z, 0, 0, 0]])  # body rotation with respect to origin might not affect pixel motion, set the last three columns to zero

            # R_c2b = np.array([[-1, 0,  0],
            #                   [ 0, 1,  0],
            #                   [ 0, 0, -1]])
            R_c2b = np.array([[-math.sqrt(2)/2, -math.sqrt(2)/2,  0],
                              [-math.sqrt(2)/2,  math.sqrt(2)/2,  0],
                              [0,                0,              -1]])# this R is causually estimated by yxj, not carefully measured.
            
            J_base2cam = np.block([[R_c2b,np.zeros((3,3))],[np.zeros((3,3)),R_c2b]])
            self.Js_hat = J_cam2img @ J_base2cam  # Js_hat = J_base2img
            """

            # new version
            R_b2c = self.R_b2c
            Js = self.base_Js
            p_s = np.array([0.058690, 0.067458, -0.053400])  # panda_EE中marker中心的坐标
            cross_mat = np.array([[0,        p_s[2], -p_s[1]],
                                  [-p_s[2],  0,       p_s[0]],
                                  [p_s[1],  -p_s[0],  0]])
            Jrot = np.block([[R_b2c, R_b2c], [np.zeros((3, 6))]])
            Jvel = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), cross_mat]])
            self.Js_hat = Js @ Jrot @ Jvel

            # real world version
            """
            fx, fy = MyConstantsSim.FX_HAT, MyConstantsSim.FY_HAT
            u, v = MyConstantsSim.U0, MyConstantsSim.V0
            z = 1
            J_cam2img = np.array([[fx/z, 0,    -u/z, 0, 0, 0],
                                  [0,    fy/z, -v/z, 0, 0, 0]])

            R_c2b = np.array([[-1, 0,  0],
                              [0,  1,  0],
                              [0,  0, -1]])
            J_base2cam = np.block([[R_c2b,np.zeros((3,3))],[np.zeros((3,3)),R_c2b]])

            p_s = np.array([0.058690, -0.067458, 0.053400])
            p_s_cross = np.array([[0, -p_s[2], p_s[1]], \
                                  [p_s[2], 0, -p_s[0]], \
                                  [-p_s[1], p_s[0], 0]])
            J_p_cross = np.block([[np.eye(3),p_s_cross],[np.zeros((3,3)),np.zeros((3,3))]])
            
            self.Js_hat = J_cam2img @ J_base2cam @ J_p_cross
            """

            # an unprecise version
            """
            self.Js_hat = np.array([[-5000, 0, -1000, -100, -100, 100],
                                    [-300, 4500, 1000, -100, -100, 100]])
            """

        if L is not None:
            if L.shape != (self.n_k, self.n_k):  # (1000, 1000)
                raise ValueError('Dimension of L should be ' + str((self.n_k, self.n_k)) + '!')
            self.L = L
        else:
            self.L = np.eye(1000) * 40000
            # raise ValueError('Matrix L should not be empty!')

        if W_hat is not None:
            if W_hat.shape != (2 * self.m, self.n_k):  # (12, 1000)
                raise ValueError('Dimension of W_hat should be ' + str((2 * self.m, self.n_k)) + '!')
            self.W_hat = W_hat
        else:
            self.W_hat = np.zeros((2*6,1000)) # initial all w are zeros
            # raise ValueError('Matrix W_hat should not be empty!')
        self.W_init_flag = False  # inf W_hat has been initialized, set the flag to True

        cfg = {'n_dim': 3,
               'n_k_per_dim': 10,
               'sigma': 1,
               'pos_restriction': np.array([[-0.35, 0.30], [0.25, 0.65], [0.40, 0.70]])}
            #    'pos_restriction': np.array([[-0.3, 0.7], [-0.3, 0.7], [0, 1]])}
        self.theta = RadialBF(cfg=cfg)
        self.theta.init_rbf_()

        self.image_space_region = ImageSpaceRegion(b=np.array([1920, 1080]))

        self.cartesian_space_region = CartesianSpaceRegion()
        self.cartesian_quat_space_region = CartesianQuatSpaceRegion()
        # self.cartesian_space_region.set_r_c(np.array([-0.01624475011413961, 0.5786721263542499, 0.30532807964440667]))  # grasping pose on the right
        # self.cartesian_space_region.set_r_c(np.array([-0.0068108842682527, 0.611158320250102, 0.5342875493162069]))  # set by yxj
        self.cartesian_space_region.set_r_c(np.array([-0.0011823860573642849, 0.43430624374805804, 0.569872105919327]))  # set by jyp | grasping pose above the second object with marker
        self.cartesian_space_region.set_c(np.array([0.05, 0.05, 0.05]).reshape(1, 3))
        self.cartesian_space_region.set_Kc(np.array([5e-5, 5e-5, 5e-5]).reshape(1, 3))

        # self.cartesian_quat_space_region.set_q_g(np.array([-0.2805967680249283, 0.6330528569977758, 0.6632800072901188, 0.2838309407825178]))  # set by yxj | grasping pose on the right
        self.cartesian_quat_space_region.set_q_g(np.array([-0.17492908847362298, 0.6884405719242297, 0.6818253503208791, 0.17479727175084528]))  # set by jyp | grasping pose above the second object with marker
        self.cartesian_quat_space_region.set_Ko(60)

        self.joint_space_region = JointSpaceRegion()
        # self.joint_space_region.add_region_multi(qc=np.array([1.48543711, -0.80891253, 1.79178384, -2.14672403, 1.74833518, 3.15085406, 2.50664708]), \
        #                                             qbound=0.50, qrbound=0.45, \
        #                                             mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
        #                                             kq=1000000, kr=10000, \
        #                                             inner=True, scale=np.ones((7, 1)))

    def kesi_x(self, x):
        return self.image_space_region.kesi_x(x.reshape(1, -1))

    def kesi_r(self, r):
        return self.cartesian_space_region.kesi_r(r.reshape(1, -1))

    def kesi_rq(self, rq):
        return self.cartesian_quat_space_region.kesi_rq_omega(rq.reshape(-1,)) / 2

    def kesi_q(self, q):
        return self.joint_space_region.kesi_q(q.reshape(1, 7))

    def get_theta(self, r):
        return self.theta.get_rbf_(r)

    def get_Js_hat(self, x, p_s=None):
        # x is the pixel coordinate on the image plane
        # p_s is the vector pointing from {NE}_origin to ArUco marker center

        # Js = self.Js_hat
        # return Js

        if p_s is None:
            return self.Js_hat
        else:
            R_b2c = self.R_b2c
            Js = self.base_Js
            Js[0, 2], Js[1, 2] = -(x[0] - MyConstantsSim.U0) / 0.5, -(x[1] - MyConstantsSim.V0) / 0.5
            cross_mat = np.array([[0,        p_s[2], -p_s[1]],
                                  [-p_s[2],  0,       p_s[0]],
                                  [p_s[1],  -p_s[0],  0]])
            Jrot = np.block([[R_b2c, R_b2c], [np.zeros((3, 6))]])
            Jvel = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), cross_mat]])

            return Js @ Jrot @ Jvel

    def get_u(self, J, d, r_t, r_o, q, x, with_vision=False, p_s=None):
        J_pinv = J.T @ np.linalg.pinv(J @ J.T)

        kesi_x = self.kesi_x(x).reshape(-1, 1)  # (2, 1)

        kesi_r = self.kesi_r(r_t.reshape(1, 3))  # (1, 3)
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
            if self.allow_adaptive:
                Js_hat = self.Js_hat
            else:
                Js_hat = self.get_Js_hat(x, p_s=p_s)
            
            u = - J_pinv @ (Js_hat.T @ kesi_x + kesi_rall + J @ kesi_q)  # normal version in paper
            # Js_hat_pinv = Js_hat.T @ np.linalg.inv(Js_hat @ Js_hat.T)  # try pseudo inverse of Js
            # u = - J_pinv @ (Js_hat_pinv @ kesi_x + kesi_rall + J_pinv.T @ kesi_q)
        else:
            u = - J_pinv @ (kesi_rall + J @ kesi_q)

        return u

    def update(self, J=None, r_t=None, r_o=None, q=None, x=None, p_s=None): # used when adaptive, if u are precise, don't use it
        if self.fa is not None:
            r = self.fa.get_pose()  # get r: including translation and rotation matrix
            q = self.fa.get_joints()  # get q: joint angles
            # split Cartesian translation and quaternion
            r_tran = r.translation
            r_quat = r.quaternion  # w, x, y, z
            # get the analytic jacobian (6*7)
            J = self.fa.get_jacobian(q)
        else:
            r_tran = r_t
            r_quat = r_o
            # J, q and x are passed through parameters

        theta = self.get_theta(r_tran).reshape(-1, 1)  # get the neuron values theta(r) (1000*1)
        if not self.W_init_flag:
            """
                Initial Method #1
            """
            # self.W_hat[:, 0] = self.Js_hat.reshape(-1,) / theta[0]
            """
                Initial Method #2
            """
            # self.W_hat = np.random.rand(12, 1000)
            """
                Initial Method #3
            """
            # for r_idx in range(self.W_hat.shape[0]):
            #     self.W_hat[r_idx, :] = (self.Js_hat.flatten()[r_idx] / np.sum(theta))
            # print("self.W_hat[r_idx, :]",self.W_hat[r_idx, :])# 一整行都是一个数？？
            # self.W_init_flag = True
            """
                Initial Method #4
                With the modified image Jacobian
            """
            # Js_hat = self.get_Js_hat(x=x, p_s=p_s)
            # for r_idx in range(self.W_hat.shape[0]):  # assigned with the same value in single line of W 
            #     self.W_hat[r_idx, :] = (Js_hat.flatten()[r_idx] / np.sum(theta))
            # self.W_init_flag = True
            """
                Initial Method #5
            """
            Js_hat_for_init = np.array([[-5000, 0, -1000, -100, -100, 100],
                                        [-300, 4500, 1000, -100, -100, 100]])
            for r_idx in range(self.W_hat.shape[0]):  # assigned with the same value in single line of W 
                self.W_hat[r_idx, :] = (Js_hat_for_init.flatten()[r_idx] / np.sum(theta))
            self.Js_hat = Js_hat_for_init
            self.W_init_flag = True
        
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        
        kesi_x = self.kesi_x(x).reshape(-1, 1)  # (2, 1)
        kesi_rt = self.kesi_r(r_tran).reshape(-1, 1)  # (3, 1)
        kesi_rq = self.kesi_rq(r_quat)  # (1, 4) @ (4, 3) = (1, 3)
        kesi_rall = np.r_[kesi_rt, kesi_rq.reshape(3, 1)]  # (6, 1)
        kesi_q = self.kesi_q(q).reshape(-1, 1)  # (7, 1)

        kesi_x_prime = np.c_[kesi_x[0] * np.eye(6), kesi_x[1] * np.eye(6)]  # (6, 12)

        dW_hat = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_rall + J_pinv.T @ kesi_q).T  # (1000, 6)
        dW_hat = dW_hat @ kesi_x_prime  # (1000, 12)

        self.W_hat = self.W_hat + dW_hat.T

        # update J_s
        temp_Js_hat = self.W_hat @ theta  # (12, 1)
        # np.c_[temp_Js_hat[:6], temp_Js_hat[6:]].T
        self.Js_hat = np.c_[temp_Js_hat[:6], temp_Js_hat[6:]].T  # (2, 6)

class JointOutputRegionControl(object):
    # def __init__(self, sim_or_real='sim', fa: FrankaArm=None) -> None:
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

        self.joint_space_region.add_region_multi(qc=np.array([1.1276347655767918, -1.0981265049638491, 2.0099641655538036, -2.1242269583241704, 2.3038052212417757, 2.816829597776607, 2.4020218102461683]), \
                                                 qbound=0.12, qrbound=0.10, \
                                                 mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
                                                 kq=1, kr=0.01, \
                                                 inner=False, scale=np.ones((7, 1)))

        self.joint_space_region.add_region_multi(qc=np.array([1.48543711, -0.80891253, 1.79178384, -2.14672403, 1.74833518, 3.15085406, 2.50664708]), \
                                                 qbound=0.50, qrbound=0.45, \
                                                 mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
                                                 kq=1000000, kr=10000, \
                                                 inner=True, scale=np.ones((7, 1)))

    def calc_manipubility(self, J_b):
        det = np.linalg.det(J_b @ J_b.T)
        return math.sqrt(np.abs(det))

    def get_dq_d_(self, q:np.ndarray, d:np.ndarray=np.zeros((7, 1)), J_sim:np.ndarray=None, time_start_this_loop=None):
        q, d = q.reshape(7,), d.reshape(7, 1)
        if self.sim_or_real == 'real':
            J = self.fa.get_jacobian(q)  # get the analytic jacobian (6*7)
        elif self.sim_or_real == 'sim':
            J = J_sim
        print('time consumption2: ', time.time() - time_start_this_loop)
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        print('time consumption3: ', time.time() - time_start_this_loop)
        N = np.eye(7) - J_pinv @ J  # get the zero space matrix (7*7)
        kesi_q = self.joint_space_region.kesi_q(q).reshape(7, 1)

        dq_d = - J_pinv @ (J @ kesi_q) + N @ np.linalg.inv(self.cd) @ d  # (7, 1)
        # dq_d = - kesi_q

        return dq_d, kesi_q

# joint space test code of my adaptive control
def test_joint_space_region_control():
    class data_collection(object):
        def __init__(self) -> None:
            self.J = np.zeros((6, 7))
            self.q = np.zeros((7,))
            self.J_ready = False
            self.q_ready = False

        def zero_jacobian_callback(self, msg):
            self.J = np.array(msg.data).reshape(6, 7)
            if not self.J_ready:
                self.J_ready = True

        def joint_angles_callback(self, msg):
            self.q = np.array(msg.data).reshape(7,)
            if not self.q_ready:
                self.q_ready = True

        def is_data_ready(self):
            return self.J_ready & self.q_ready

    data_c = data_collection()
    controller = JointOutputRegionControl(sim_or_real='sim', fa=None)
    
    nh_ = rospy.init_node('joint_space_region_testbench', anonymous=True)
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    sub_q_ = rospy.Subscriber('/gazebo_sim/joint_angles', Float64MultiArray, data_c.joint_angles_callback, queue_size=1)
    rate_ = rospy.Rate(30)

    dq_d_list = []
    kesi_q_list = []
    dist_list = []
    q_and_manipubility_list = np.zeros((0, 8))

    time_start = time.time()

    while not rospy.is_shutdown():
        if data_c.is_data_ready():
            time_start_this_loop = time.time()
            q_and_m = np.zeros((1, 8))
            q_and_m[0, :7] = data_c.q
            q_and_m[0, 7] = controller.calc_manipubility(data_c.J)
            dist = np.linalg.norm((np.array([1.48543711, -0.80891253, 1.79178384, -2.14672403, 1.74833518, 3.15085406, 2.50664708]) - data_c.q), ord=2)
            dist_list.append(dist)
            q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)
            print('time consumption1: ', time.time() - time_start_this_loop)
            dq_d_, kesi_q = controller.get_dq_d_(q=data_c.q, d=np.zeros((7, 1)), J_sim=data_c.J, time_start_this_loop=time_start_this_loop)
            dq_d_list.append(dq_d_.reshape(7,).tolist())
            kesi_q_list.append(kesi_q.reshape(7,).tolist())
            msg = Float64MultiArray()
            msg.data = dq_d_.reshape(7,)
            pub_.publish(msg)
            print('time consumption4: ', time.time() - time_start_this_loop)

            if time.time() - time_start >= 25.0:
                break

        rate_.sleep()

    np.save('./data/0520/q_and_m.npy', q_and_manipubility_list)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(dq_d_list)
    plt.legend(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
    plt.subplot(2, 2, 2)
    plt.plot(kesi_q_list)
    plt.legend(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
    plt.subplot(2, 2, 3)
    plt.plot(dist_list)
    plt.subplot(2, 2, 4)
    plt.plot(q_and_manipubility_list[:, 7])
    plt.show()
    info = {'dq_d': dq_d_list, \
            'kesi_q': kesi_q_list, \
            'dist': dist_list, \
            'q_and_manip': q_and_manipubility_list}
    with open('./data/0520/q_and_manip.pkl', 'wb') as f:
        pickle.dump(info, f)

# Cartesian space test code of my adaptive control
# -0.4274491954570557, 0.17649338322602143, 0.0877387993047109, 0.9941285663016644, -0.0020305968914211114, 0.10663487821459938, 0.018123963264730075
def test_cartesian_joint_space_region_control():
    class data_collection(object):
        def __init__(self) -> None:
            self.J = np.zeros((6, 7))
            self.q = np.zeros((7,))
            self.trans = np.zeros((3,))
            self.quat = np.zeros((4,))
            self.J_ready = False
            self.q_ready = False
            self.pose_ready = False

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

        def is_data_ready(self):
            return self.J_ready & self.q_ready & self.pose_ready

    data_c = data_collection()
    controller_j = JointSpaceRegion()
    controller_r = CartesianSpaceRegion()
    controller_rq = CartesianQuatSpaceRegion()

    # init control scheme
    # controller_r.set_r_c(np.array([-0.4274491954570557, 0.17649338322602143, 0.0877387993047109]).reshape(1, 3))
    # controller_r.set_r_c(np.array([0.30705422269100857, -7.524700079232461e-06, 0.4870834722065375]))  # starting point
    # controller_r.set_r_c(np.array([-0.01624475011413961, 0.5786721263542499, 0.30532807964440667]))  # grasping pose on the right
    controller_r.set_r_c(np.array([-0.00046233226943146605, 0.4572876569970451, 0.4897459710192101]))  # grasping pose above the second object with marker
    controller_r.set_c(np.array([0.01, 0.01, 0.01]).reshape(1, 3))
    controller_r.set_Kc(np.array([1e-7, 1e-7, 1e-7]).reshape(1, 3))

    # controller_rq.set_q_g(np.array([0.018123963264730075, 0.9941285663016644, -0.0020305968914211114, 0.10663487821459938]))
    # controller_rq.set_q_g(np.array([0.0004200682001275777, 0.9028032461428314, -0.00021468888469159875, -0.4300479986956025]))  # rot_Y(-90)
    # controller_rq.set_q_g(np.array([0.0005740008069370812, 0.708744905485196, -0.7054320813712253, 0.00603014709058085]))  # rot_Z(+90)
    # controller_rq.set_q_g(np.array([0.5396668336145884, -0.841829320383993, 0.00250734600652218, -0.008486521399793301]))  # rot_X(+90)
    # controller_rq.set_q_g(np.array([-0.2805967680249283, 0.6330528569977758, 0.6632800072901188, 0.2838309407825178]))  # grasping pose on the right
    controller_rq.set_q_g(np.array([-0.16181711157880402, 0.6981663251376432, 0.6773967396126247, 0.16584134878475798]))  # grasping pose above the second object with marker
    controller_rq.set_Ko(15)
    
    nh_ = rospy.init_node('cartesian_joint_space_region_testbench', anonymous=True)
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    sub_q_ = rospy.Subscriber('/gazebo_sim/joint_angles', Float64MultiArray, data_c.joint_angles_callback, queue_size=1)
    sub_ee_pose_ = rospy.Subscriber('/gazebo_sim/ee_pose', Float64MultiArray, data_c.pose_callback, queue_size=1)
    rate_ = rospy.Rate(100)

    f_quat_list, p_quat_list, quat_list = [], [], []
    dist_list = []
    kesi_rall_list = []
    time_list = []
    position_list = []
    q_and_manipubility_list = np.zeros((0, 8))
    start_time = time.time()

    # update control scheme
    while not rospy.is_shutdown():
        if data_c.is_data_ready():
            q_and_m = np.zeros((1, 8))
            q_and_m[0, :7] = data_c.q
            q_and_m[0, 7] = math.sqrt(np.abs(np.linalg.det(data_c.J @ data_c.J.T)))
            dist = np.linalg.norm((np.array([1.48543711, -0.80891253, 1.79178384, -2.14672403, 1.74833518, 3.15085406, 2.50664708]) - data_c.q), ord=2)
            dist_list.append(dist)
            q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)

            kesi_r = controller_r.kesi_r(data_c.trans.reshape(1, 3))  # (1, 3)
            # kesi_rq = controller_rq.kesi_rq(data_c.quat) / 50  # (1, 3)
            if controller_rq.fo(Quat(data_c.quat)) <= 0:
                kesi_rq = np.zeros((1, 3))
            else:
                kesi_rq = controller_rq.kesi_rq_omega(data_c.quat) / 2 # (1, 3)
            kesi_q = controller_j.kesi_q(data_c.q).reshape(7, 1)  # (7, 1)
            kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)

            # convertion between spatial Twist and body Twist
            rotMat = R.from_quat(data_c.quat[[1, 2, 3, 0]]).as_matrix()
            p = data_c.trans
            pMat = np.array([[0, -p[2], p[1]], \
                             [p[2], 0, -p[0]], \
                             [-p[1], p[0], 0]])
            T = np.block([[rotMat, np.zeros((3, 3))], [pMat @ rotMat, rotMat]])

            J = data_c.J
            J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
            
            # dq_d_ = -J_pinv @ (kesi_rall + J @ kesi_q)
            dq_d_ = -J_pinv @ kesi_rall
            msg = Float64MultiArray()
            msg.data = dq_d_.reshape(7,)
            pub_.publish(msg)

            f_quat_list.append(controller_rq.fo(Quat(data_c.quat)))
            p_quat_list.append(controller_rq.Po(Quat(data_c.quat)))
            quat_list.append(data_c.quat.tolist())
            kesi_rall_list.append(kesi_rall)
            position_list.append(p)
            time_list.append(time.time()-start_time)
            if time.time() - start_time >= 30.0:
                break

            # print(kesi_r)
            # print(kesi_rq)
        
        rate_.sleep()
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time_list, f_quat_list, color='b',label = 'f_quat')
    plt.plot(time_list, p_quat_list, color='r',label = 'p_quat')
    plt.title('f and p for quaternion')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(time_list, quat_list)
    plt.title('quaternion vs time')

    plt.figure()
    ax1 = plt.axes(projection='3d')
    position_array  = np.array(position_list)
    ax1.plot3D(position_array[:,0],position_array[:,1],position_array[:,2],label='traj')
    ax1.scatter(position_array[0,0],position_array[0,1],position_array[0,2],c='r',label='initial')
    ax1.scatter(position_array[200,0],position_array[200,1],position_array[200,2],c='b',label='t=5s')
    ax1.scatter(-0.01624475011413961, 0.5786721263542499, 0.30532807964440667,c='g',label='goal region center')
    ax1.legend()
    ax1.set_xlabel('x/m')
    ax1.set_ylabel('y/m')
    ax1.set_zlabel('z/m')
    plt.title('executed trajectory 3D')

    plt.figure()
    plt.plot(time_list,np.reshape(kesi_rall_list,(np.shape(time_list)[0],-1)),label = 'kesi')
    plt.legend()
    plt.title('kesi for 6 dimensions')


    plt.show()
    info = {'dist': dist_list, \
            'q_and_manip': q_and_manipubility_list}
    with open('./data/0528/q_and_manip.pkl', 'wb') as f:
        pickle.dump(info, f)

# 0616 yxj
def test_vision_joint_space_region_control():
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
            self.x1 = np.zeros((2,))
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

        def is_data_without_vision_ready(self):
            return self.J_ready & self.q_ready & self.pose_ready

        def is_data_with_vision_ready(self):
            return self.J_ready & self.q_ready & self.pose_ready & self.vision_1_ready & self.vision_2_ready

    data_c = data_collection()
    controller_j = JointSpaceRegion()
    controller_r = CartesianSpaceRegion()
    controller_rq = CartesianQuatSpaceRegion()
    controller_x = ImageSpaceRegion(b=np.array([1920,1080]))
    
    nh_ = rospy.init_node('cartesian_joint_space_region_testbench', anonymous=True)
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    sub_q_ = rospy.Subscriber('/gazebo_sim/joint_angles', Float64MultiArray, data_c.joint_angles_callback, queue_size=1)
    sub_ee_pose_ = rospy.Subscriber('/gazebo_sim/ee_pose', Float64MultiArray, data_c.pose_callback, queue_size=1)
    sub_vision_1_ = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, data_c.vision_1_callback, queue_size=1)
    sub_vision_2_ = rospy.Subscriber('/aruco_simple/pixel2', PointStamped, data_c.vision_2_callback, queue_size=1)
    rate_ = rospy.Rate(100)

    # move to the place where the camera can see the ee aruco marker
    target_joint = np.array([0.79083, -0.78804, 0.35310, -2.43173, 0.51854, 2.02235, 0.12722])  # exactly on the right
    # target_joint = np.array([2.09541, 0.065632, 0.10139, -1.55448, 0.16644, 1.65392, 0.64225])  # on the right behind
    target_quat = np.array([-0.16182, 0.69817, 0.67740, 0.16584])  # grasping pose
    time_start1 = time.time()
    while not rospy.is_shutdown():
        if data_c.is_data_without_vision_ready():
            rospy.loginfo('Data with vision info has been prepared yet!')
            time_start_this_loop = time.time()
            q = data_c.q
            # print('q',q)
            # print('target',target_joint)
            kp = 1
            input = kp*(target_joint-q)
            msg = Float64MultiArray()
            msg.data = input.reshape(7,)
            pub_.publish(msg)
            # print('time consumption4: ', time.time() - time_start_this_loop)

            if time.time() - time_start1 >= 6.0:
                msg.data = [0,0,0,0,0,0,0]
                pub_.publish(msg)
                break

    # init control scheme
    controller_rq.set_q_g(target_quat)
    controller_rq.set_Ko(100)
    print('Cartesian quaternion region is set!')
    if data_c.is_data_without_vision_ready():
        controller_x.set_x_d(data_c.x2 - np.array([0, 100]))
        # controller_x.set_b(50)
        controller_x.set_Kv(0.2)
        print('Vision region is set!')


    f_list, p_list, kesi_x_list, pixel_1_list, pixel_2_list, time_list=[], [], [], [], [], []
    q_and_manipubility_list = np.zeros((0, 8))

    camera_intrinsic = np.array([3759.66467, 0.0, 960.5, 0.0, 3759.66467, 540.5, 0.0, 0.0, 1.0]).reshape((3,3))
    fx = camera_intrinsic[0,0]
    fy = camera_intrinsic[1,1]
    u0 = camera_intrinsic[0,2]
    v0 = camera_intrinsic[1,2]    
    
    R_c2b = np.array([[-math.sqrt(2)/2, -math.sqrt(2)/2,  0],
    [-math.sqrt(2)/2, math.sqrt(2)/2,  0],
    [0,  0, -1]])# this R is causually estimated by yxj, not carefully measured.
    depth = 1

    # print('Js',Js)

    start_time = time.time() 
    # update control scheme
    while not rospy.is_shutdown():
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = data_c.q
        q_and_m[0, 7] = math.sqrt(np.abs(np.linalg.det(data_c.J @ data_c.J.T)))

        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)

        # get all kesi from controllers
        if controller_rq.fo(Quat(data_c.quat)) <= 0:
            kesi_rq = np.zeros((1, 3))
        else:
            kesi_rq = controller_rq.kesi_rq_omega(data_c.quat) / 2 # (1, 3)
        kesi_r = np.zeros((1, 3))  # (1, 3)
        kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)

        kesi_x = controller_x.kesi_x(data_c.x1)
        kesi_x = kesi_x.reshape((2,1))

        J = data_c.J
        J_pos = J[:3,:] # 3x7
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7x6)
        J_pos_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T) # 7x3
        
        u = data_c.x1[0]-u0
        v = data_c.x1[1]-v0
        Js = np.array([[fx/depth,0,u/depth],[0,fy/depth,v/depth]]) @ R_c2b # 2x3

        # print("kesi_x",kesi_x)
        # print("Js[:2,:].T",Js[:2,:].T)
        # print("J_pos_pinv",J_pos_pinv)

        dq_d_ = -J_pos_pinv @ (Js.T @ kesi_x) - J_pinv @ kesi_rall
        msg = Float64MultiArray()
        msg.data = dq_d_.reshape(7,)
        pub_.publish(msg)

        # logging
        time_list.append(time.time()-start_time)
        pixel_1_list.append(data_c.x1)
        pixel_2_list.append(data_c.x2)
        f_list.append(controller_x.fv(data_c.x1))
        p_list.append(controller_x.Pv(data_c.x1))
        kesi_x_list.append(kesi_x.reshape(2,))

        if time.time() - start_time >= 30.0:
            break

            # print(kesi_r)
            # print(kesi_rq)
        
        rate_.sleep()

    pre_traj = './data/0618/'
    
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

    plt.show()
    info = {'f_list': f_list, \
            'p_list': p_list, \
            'kesi_x_list': kesi_x_list, \
            'pixel_1_list': pixel_1_list, \
            'pixel_2_list': pixel_2_list, \
            'time_list': time_list, \
            'q_and_manipubility_list':q_and_manipubility_list}
    with open('./data/0618/data.pkl', 'wb') as f:
        pickle.dump(info, f)

# 0623 yxj
def test_adaptive_region_control(allow_update=False):
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

        def is_data_with_vision_1_ready(self):
            return self.J_ready & self.q_ready & self.pose_ready & self.vision_1_ready & self.vision_2_ready

    # launch Tensorboard
    writer = SummaryWriter(log_dir='./data/0705/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), flush_secs=5)
    
    desired_position_bias = -np.array([240, 160])

    data_c = data_collection()
    controller_adaptive = AdaptiveRegionControllerSim(allow_adaptive=allow_update)
    
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
            controller_adaptive.image_space_region.set_x_d(data_c.x2 + desired_position_bias)
            # print("1111",controller_adaptive.image_space_region.x_d)
            # controller_adaptive.image_space_region.set_b(50)
            controller_adaptive.image_space_region.set_Kv(np.array([[0.2, 0.1]]))
            print('vision region is set!')
            break

    f_list, p_list, kesi_x_list, pixel_1_list, pixel_2_list, time_list=[], [], [], [], [], []
    q_and_manipubility_list = np.zeros((0, 8))
    f_quat_list,p_quat_list,quat_list,kesi_rall_list,position_list = [],[],[],[],[]

    start_time = time.time()
    Js_array = np.zeros((0, 12))
    # update control scheme
    """
        Task1: Set Point Control
    """
    cnt = 0
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

        """
            unless the marker attached to gripper is seen
            donnot update Js and exclude it from calculating dq_d_
        """
        # 示例：计算p_s
        ee_pose_quat = data_c.quat[[1, 2, 3, 0]]
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        p_s_in_panda_EE = np.array([0.058690, 0.067458, -0.053400])
        p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)
        if data_c.is_data_with_vision_1_ready():
            controller_adaptive.cartesian_quat_space_region.set_Ko(0)
            dq_d_ = controller_adaptive.get_u(J, d, data_c.trans, data_c.quat, data_c.q, data_c.x1, with_vision=True, p_s=p_s.reshape(-1,))
            if allow_update:
                controller_adaptive.update(J, data_c.trans, data_c.quat, data_c.q, data_c.x1, p_s=p_s.reshape(-1,))
                Js_array = np.concatenate((Js_array, controller_adaptive.Js_hat.reshape(1, 12)))
        else:
            dq_d_ = controller_adaptive.get_u(J, d, data_c.trans, data_c.quat, data_c.q, data_c.x1, with_vision=False, p_s=p_s.reshape(-1,))
        print('Js: ', controller_adaptive.get_Js_hat(None, None))

        msg = Float64MultiArray()
        msg.data = dq_d_.reshape(7,)
        pub_.publish(msg)

        # logging
        time_list.append(time.time()-start_time)
        pixel_1_list.append(data_c.x1)
        pixel_2_list.append(data_c.x2 + desired_position_bias)
        f_list.append(controller_adaptive.image_space_region.fv(data_c.x1))
        p_list.append(controller_adaptive.image_space_region.Pv(data_c.x1).squeeze())
        kesi_x_list.append(controller_adaptive.image_space_region.kesi_x(data_c.x1).reshape((-1,)))

        f_quat_list.append(controller_adaptive.cartesian_quat_space_region.fo(Quat(data_c.quat)))
        p_quat_list.append(controller_adaptive.cartesian_quat_space_region.Po(Quat(data_c.quat)))
        quat_list.append(data_c.quat.tolist())
        kesi_rall_list.append(controller_adaptive.kesi_rall)
        position_list.append(data_c.trans)

        # log to Tensorboard
        # track the convergence of Js
        if data_c.is_data_with_vision_1_ready():
            if allow_update:
                Js = controller_adaptive.Js_hat
            else:
                Js = controller_adaptive.get_Js_hat(x=data_c.x1, p_s=p_s)
            log_dict = {}
            for col in range(6):
                log_dict[',{0})'.format(col + 1)] = Js[0, col]
            writer.add_scalars('Js_r(0', log_dict, global_step=cnt)
            log_dict.clear()
            for col in range(6):
                log_dict[',{0})'.format(col + 1)] = Js[1, col]
            writer.add_scalars('Js_r(1', log_dict, global_step=cnt)

            # track the tracking task performance
            log_dir = {'goal_x': data_c.x2.squeeze()[0] + desired_position_bias[0],
                        'pos_x': data_c.x1[0]}
            writer.add_scalars('track_x', log_dir, global_step=cnt)
            log_dir = {'goal_y': data_c.x2.squeeze()[1] + desired_position_bias[1],
                        'pos_y': data_c.x1[1]}
            writer.add_scalars('track_y', log_dir, global_step=cnt)

            # track the convergence of weight matrix W
            if allow_update:
                fig = plt.figure()
                plt.imshow((controller_adaptive.W_hat - np.min(controller_adaptive.W_hat, axis=1).reshape(-1, 1)) / (np.max(controller_adaptive.W_hat, axis=1).reshape(-1, 1) - np.min(controller_adaptive.W_hat, axis=1).reshape(-1, 1)))
                writer.add_figure('matrix W', fig, global_step=cnt)

        if time.time() - start_time >= 30.0:
            """
            plt.imshow((controller_adaptive.W_hat - np.min(controller_adaptive.W_hat, axis=1).reshape(-1, 1)) / (np.max(controller_adaptive.W_hat, axis=1).reshape(-1, 1) - np.min(controller_adaptive.W_hat, axis=1).reshape(-1, 1)))
            plt.show()

            plt.figure()
            for row in range(3):
                for col in range(4):
                    index = row * 4 + col + 1
                    plt.subplot(3, 4, index)
                    plt.plot(Js_array[:, index - 1])
            plt.show()
            """
            break
        
        cnt = cnt + 1
        rate_.sleep()

    """
        Task2: Spring-like trajectory tracking control
    """
    """
    class spring_traj(object):
        def __init__(self, x0:np.ndarray, pos_orient:np.ndarray, omega:float) -> None:
            self.x0 = x0
            self.pos_orient = pos_orient
            self.omega = omega

        def get_point(self, time:float):
            return self.x0 + self.pos_orient * math.sin(self.omega * time)

    print('Started to track spring like trajectory!')

    start_time = time.time()  # restart timer
    cnt = 0
    # define the spring like trajectory
    pos_orient = np.array([150, -100])
    x0 = data_c.x2 + desired_position_bias + np.array([400, 300])
    period = 20
    omega = 2 * math.pi / period
    traj = spring_traj(x0=x0,
                        pos_orient=pos_orient,
                        omega=omega)
    while not rospy.is_shutdown():
        track_goal = traj.get_point(time=time.time() - start_time).reshape(1, 2)
        controller_adaptive.image_space_region.set_x_d(track_goal)

        # 示例: 计算p_s
        ee_pose_quat = data_c.quat[[1, 2, 3, 0]]
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        p_s_in_panda_EE = np.array([0.058690, 0.067458, -0.053400])
        p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)
        dq_d_ = controller_adaptive.get_u(J, d, data_c.trans, data_c.quat, data_c.q, data_c.x1, with_vision=True, p_s=p_s.reshape(-1,))
        if allow_update:
            controller_adaptive.update(J, data_c.trans, data_c.quat, data_c.q, data_c.x1, p_s=p_s.reshape(-1,))

        # send joint velocity command to Franka
        msg = Float64MultiArray()
        msg.data = dq_d_.reshape(7,)
        pub_.publish(msg)

        # log to Tensorboard
        if allow_update:
            # track the convergence of Js
            Js = controller_adaptive.Js_hat
            log_dict = {}
            for col in range(6):
                log_dict['Js_{0}_{1}'.format(0, col + 1)] = Js[0, col]
            writer.add_scalars('Js_r0', log_dict, global_step=cnt)
            log_dict.clear()
            for col in range(6):
                log_dict['Js_{0}_{1}'.format(1, col + 1)] = Js[1, col]
            writer.add_scalars('Js_r1', log_dict, global_step=cnt)

            # track the tracking task performance
            log_dir = {'goal_x': track_goal.squeeze()[0],
                       'pos_x': data_c.x1[0]}
            writer.add_scalars('track_x', log_dir, global_step=cnt)
            log_dir = {'goal_y': track_goal.squeeze()[1],
                       'pos_y': data_c.x1[1]}
            writer.add_scalars('track_y', log_dir, global_step=cnt)

            # track the convergence of weight matrix W
            fig = plt.figure()
            plt.imshow((controller_adaptive.W_hat - np.min(controller_adaptive.W_hat, axis=1).reshape(-1, 1)) / (np.max(controller_adaptive.W_hat, axis=1).reshape(-1, 1) - np.min(controller_adaptive.W_hat, axis=1).reshape(-1, 1)))
            writer.add_figure('matrix W', fig, global_step=cnt)

        if cnt >= 1000:
            break
        cnt = cnt + 1
        rate_.sleep()
    """

    """
        Task3: Circular trajectory tracking control
    """
    """
    class circle_traj(object):
        def __init__(self, center:np.ndarray, radius:int, omega:float) -> None:
            self.center = center
            self.radius = radius
            self.omega = omega
            
        def get_point(self, time:float):
            u = self.center[0] + self.radius * math.cos(self.omega * time)
            v = self.center[1] + self.radius * math.sin(self.omega * time)

            return np.array([u, v])

    print('Started to track circular trajectory!')
    print('x2: ', data_c.x2)
    start_time = time.time()  # restart timer
    cnt = 0
    Js_array = np.zeros((0, 12))
    while not rospy.is_shutdown():
        # define the circular trajectory
        radius = 300
        center = data_c.x2 - np.array([radius, 0])
        traj = circle_traj(center=center,
                           radius=radius,
                           omega=2 * math.pi / 25)
        
        track_goal = traj.get_point(time=time.time() - start_time).reshape(1, 2)
        controller_adaptive.image_space_region.set_x_d(track_goal)

        # 示例: 计算p_s
        ee_pose_quat = data_c.quat[[1, 2, 3, 0]]
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        p_s_in_panda_EE = np.array([0.058690, 0.067458, -0.053400])
        p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)
        dq_d_ = controller_adaptive.get_u(J, d, data_c.trans, data_c.quat, data_c.q, data_c.x1, with_vision=True, p_s=p_s.reshape(-1,))
        if allow_update:
            controller_adaptive.update(J, data_c.trans, data_c.quat, data_c.q, data_c.x1, p_s=p_s.reshape(-1,))
            print('Js: ', controller_adaptive.Js_hat)

        cnt = cnt + 1
        if cnt % 20 == 0:
            Js_array = np.concatenate((Js_array, controller_adaptive.Js_hat.reshape(1, 12)), axis=0)
        if cnt % 5000 == 0:
            pdb.set_trace()

        msg = Float64MultiArray()
        msg.data = dq_d_.reshape(7,)
        pub_.publish(msg)

        rate_.sleep()
    """

    # close Tensorboard
    writer.close()
    pre_traj = './data/0705/'
    
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
            'position_list':position_list, \
            'adaptive_weight_matrix':controller_adaptive.W_hat}
    with open('./data/0703/data_withno_adaptive.pkl', 'wb') as f:
        pickle.dump(info, f)
        pass


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

def plot_figures2():
    import pickle
    with open('./data/0521/q_and_manip.pkl', 'rb') as f:
        data1 = pickle.load(f)
    # with open('./data/0521/q_and_manip.pkl', 'rb') as f:
    #     data2 = pickle.load(f)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(data1['q_and_manip'][:, :7], color='r')
    # plt.plot(data2['q_and_manip'][:, :7], color='b')
    plt.title('joint positions')
    # plt.legend(['with jsr', 'with no jsr'])
    plt.subplot(1, 2, 2)
    plt.plot(data1['q_and_manip'][:, 7], color='r')
    # plt.plot(data2['q_and_manip'][:, 7], color='b')
    plt.title('manipubility')
    # plt.legend(['with jsr', 'with no jsr'])
    plt.savefig('./data/0521/test_Cartesian_space_region_0527.png', dpi=600)
    plt.show()

def plot_figures3():
    import pickle
    with open('./data/0703/data_with_adaptive.pkl', 'rb') as f:
        data1 = pickle.load(f)
    with open('./data/0703/data_withno_adaptive.pkl', 'rb') as f:
        data2 = pickle.load(f)
    desired_position_bias = np.array([240, 160])
    plt.figure()
    plt.subplot(1, 2, 1)
    pixel_1_list, pixel_2_list = data1['pixel_1_list'], data1['pixel_2_list']
    pdb.set_trace()
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='b', alpha=0.75)
    plt.scatter(pixel_2_list[0][0] - desired_position_bias[0], pixel_2_list[0][1] - desired_position_bias[1],color='b',label = 'desired position', alpha=0.75)
    
    # plot pixels every extract_interval
    extract_interval = 100
    extract_index = np.arange(0, len(pixel_1_list), extract_interval)
    extract_point = np.zeros((0, 2))
    for index in extract_index:
        extract_point = np.concatenate((extract_point, pixel_1_list[index].reshape(1, 2)), axis=0)
    plt.scatter(extract_point[:, 0], extract_point[:, 1], marker='x', color='b', alpha=0.5)

    pixel_1_list, pixel_2_list = data2['pixel_1_list'], data2['pixel_2_list']
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='r', alpha=0.75)
    plt.scatter(pixel_2_list[0][0] - desired_position_bias[0], pixel_2_list[0][1] - desired_position_bias[1],color='r',label = 'desired position', alpha=0.75)

    # plot pixels every extract_interval
    extract_interval = 100
    extract_index = np.arange(0, len(pixel_1_list), extract_interval)
    extract_point = np.zeros((0, 2))
    for index in extract_index:
        extract_point = np.concatenate((extract_point, pixel_1_list[index].reshape(1, 2)), axis=0)
    plt.scatter(extract_point[:, 0], extract_point[:, 1], marker='x', color='r', alpha=0.5)

    plt.title('vision trajectory')
    plt.legend(['with adaptive', 'with no adaptive'])
    plt.savefig('./data/0703/test_Cartesian_space_region.png', dpi=600)
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

    # test_joint_space_region_control()
    # test_cartesian_joint_space_region_control()
    # plot_figures2()
    
    # yxj 20220618
    # test_vision_joint_space_region_control()

    # yxj 20220623
    # nh_ = rospy.init_node('joint_space_region_testbench', anonymous=True)
    test_adaptive_region_control(allow_update=False)
    # plot_figures3()
