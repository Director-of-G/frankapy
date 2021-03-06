"""
    my_adaptive_control.py, 2022-05-01
    Copyright 2022 IRM Lab. All rights reserved.
"""
from cProfile import label
import collections
from multiprocessing.dummy import Pool
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

import cv2
import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import numpy as np

from my_adaptive_control import ImageSpaceRegion
# from gazebo.my_test_jyp import compute_R_c2i
from calculate_dW_hat import calculate_dW_hat

from multiprocessing import Process, Lock
import psutil

from autolab_core import RigidTransform

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
    # CARTESIAN_CENTER = np.array([0.2068108842682527, 0.611158320250102, 0.1342875493162069])
    # CARTESIAN_CENTER = np.array([-0.0068108842682527, 0.611158320250102, 0.1342875493162069])
    # CARTESIAN_CENTER = np.array([0.17, 0.55, 0.20])
    CARTESIAN_CENTER = np.array([0.09343, 0.56261, 0.20943])
    # CARTESIAN_SIDE_LENGTH = np.array([0.05, 0.05, 0.1]).reshape(1, 3)
    # CARTESIAN_SIDE_LENGTH = np.array([0.21634, 0.14524, 0.12755]).reshape(1, 3)
    CARTESIAN_SIDE_LENGTH = np.array([0.18634, 0.11524, 0.02]).reshape(1, 3)
    # CARTESIAN_KC = np.array([2e-4, 2e-4, 1e-6]).reshape(1, 3) # 0723 good param!!!
    CARTESIAN_KC = np.array([0,0,0]).reshape(1, 3)
    QUATERNION_QG = np.array([-0.2805967680249283, 0.6330528569977758, 0.6632800072901188, 0.2838309407825178])
    QUATERNION_KO = 0
    QUATERNION_RANGE_COEFF = 15
    IMG_W = 1440
    IMG_H = 1080
    IMG_WH = np.array([1440,1080])
    KESI_X_SCALE = 0.25
    CAM_HEIGHT = 1.38
    # JS_HAT_FOR_INIT = np.array([[0, 0, -1000, -100, -100, 100], \
    #                             [-300, 4500, 1000, -100, -100, 100]])
    JS_HAT_FOR_INIT = np.array([[-1000, 0, -0, -100, -100, 100], \
                                [-300, 4500, 0, -100, -100, 100]])

def MatrixMultiplication(params):
    L, theta, Js_hat, kesi_x, kesi_rall, J_pinv, kesi_q, kesi_x_prime = \
                params['L'], \
                params['theta'], \
                params['Js_hat'], \
                params['kesi_x'], \
                params['kesi_rall'], \
                params['J_pinv'], \
                params['kesi_q'], \
                params['kesi_x_prime']
    dW_hat = - L @ theta @ (Js_hat.T @ kesi_x + kesi_rall + J_pinv.T @ kesi_q).T
    dW_hat = dW_hat @ kesi_x_prime

    return dW_hat

class ImageSpaceRegion(object):
    def __init__(self, x_d=None, b=None, Kv=None) -> None:
        # x_d , b, Kv
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
        # print('Kv: ', self.Kv)
        return - self.Kv * np.minimum(0, self.fv(x)) * partial_fv

class CartesianSpaceRegion(object):
    def __init__(self, r_c=None, c=None, Kc=None) -> None:
        """
            r_c is the desired Cartesian configuration, which is [x, y, z, r, p, y].
            r???[-pi, pi], p???[-pi/2, pi/2], y???[-pi, pi], which is euler angle in the order of 'XYZ'
        """
        # r_c , c , Kc
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
    def __init__(self, q_g:np.ndarray=None, Ko=1, orient_range_coeff=1) -> None:
        # q_g, Ko, 
        self.q_g = Quat(q_g)  # Quat
        self.q_diff = Quat()  # Quat
        self.Ko = Ko  # float
        self.orient_range_coeff = orient_range_coeff

    def set_q_g(self, q_g: np.ndarray):
        self.q_g = Quat(q_g)  # Quat

    def set_orient_range(self, coeff):
        self.orient_range_coeff = coeff

    def set_Ko(self, Ko):
        self.Ko = Ko  # float
    
    def fo(self, q:Quat, return_diff=False):
        q_unit = q.unit_()
        q_g_unit = self.q_g.unit_()
        self.q_diff = q_unit.dq_(q_g_unit).unit_()
        # self.q_diff = q_g_unit.dq_(q_unit)
        if return_diff:
            return self.orient_range_coeff * self.q_diff.logarithm_(return_norm=True) - 1, self.q_diff
        else:
            return self.orient_range_coeff * self.q_diff.logarithm_(return_norm=True) - 1

    def in_region(self, q: np.ndarray):
        q = Quat(q)
        fo = self.fo(q)
        return (fo <= 0)

    def Po(self, q:Quat):
        fo = self.fo(q)

        return 0.5 * self.Ko * (np.maximum(0, fo)) ** 2

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

    # deprecated
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

        return 0.1 *(partial_P_q @ J_rot).reshape(1, -1)

    def kesi_rq_omega(self, q:np.ndarray):
        q, q_g = Quat(q).unit_(), self.q_g.unit_()
        q_diff = q.dq_(q_g)

        axis, theta = q_diff.axis_angle_(split=True)
        axis_normalized = axis / np.linalg.norm(axis, ord=2)
        if theta > math.pi:
            theta = - (2 * math.pi - theta)
        return self.Ko * theta * axis_normalized.reshape(1, 3)
        
class JointSpaceRegion(object):
    def __init__(self) -> None:
        self.multi_kq = np.zeros((1, 0))
        self.multi_kr = np.zeros((1, 0))
        self.multi_qc = np.zeros((7, 0))
        self.multi_scale = np.zeros((7, 0))
        self.multi_mask = np.zeros((7, 0))
        self.multi_qbound = np.zeros((1, 0))
        self.multi_qrbound = np.zeros((1, 0))
        self.multi_inout = np.zeros((1, 0), dtype=np.bool8)  # in => 1, out => (-1)

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
        n_multi = self.multi_qc.shape[1]  # (1, n_single) and (7, n_multi)

        q = q.reshape(7, 1)
        q_scale_multi = q * self.multi_scale  # (7, n_multi)
        fq_multi = np.sum(((q_scale_multi - self.multi_qc) ** 2) * self.multi_mask, axis=0) - self.multi_qbound ** 2  # (1, n_multi)
        fqr_multi = np.sum(((q_scale_multi - self.multi_qc) ** 2) * self.multi_mask, axis=0) - self.multi_qrbound ** 2  # (1, n_multi)
        fq_multi = fq_multi * self.multi_inout  # (1, n_multi)
        fqr_multi = fqr_multi * self.multi_inout  # (1, n_multi)

        return fq_multi.reshape(1, n_multi), fqr_multi.reshape(1, n_multi)

    def in_region(self, q):
        fq_multi, _ = self.fq(q)

        return (fq_multi <= 0)

    def Ps(self, q):
        fq_multi, fqr_multi = self.fq(q)
        Ps = 0
        Ps = Ps + np.sum(0.5 * self.multi_kq * (np.minimum(0, fq_multi)) ** 2)
        Ps = Ps + np.sum(0.5 * self.multi_kr * (np.minimum(0, fqr_multi)) ** 2)

        return Ps

    def kesi_q(self, q):
        fq_multi, fqr_multi = self.fq(q)

        q = q.reshape(7, 1)
        q_scale_multi = q * self.multi_scale  # (7, n_multi)
        not_in_region_mask = (fq_multi < 0).repeat(7, 0)  # (7, n_multi)  [IMPORTANT] 
        partial_f_q = (2 * (q_scale_multi - self.multi_qc) * self.multi_inout * self.multi_mask * not_in_region_mask) # (7, n_multi)

        not_in_r_region_mask = (fqr_multi < 0).repeat(7, 0)  # (7, n_multi)  [IMPORTANT] 
        partial_fr_q = (2 * (q_scale_multi - self.multi_qc) * self.multi_inout * self.multi_mask * not_in_r_region_mask) # (7, n_multi)

        # print('partial_f_q\n',partial_f_q)
        # print('partial_fr_q\n',partial_fr_q)

        kesi_q = np.zeros((1, 7))
        # print('kesi_q_0',kesi_q)
        kesi_q = kesi_q + np.sum(self.multi_kq * np.minimum(0, fq_multi) * partial_f_q ,axis=1)
        # print('kesi_q_1',kesi_q)
        kesi_q = kesi_q + np.sum(self.multi_kr * np.minimum(0, fqr_multi) * partial_fr_q,axis=1)
        # print('kesi_q',kesi_q)
        # print('----------------------------')

        return kesi_q.reshape(1, -1)


class AdaptiveRegionController(object):
    """
        @ Class: AdaptiveRegionController
        @ Function: copied and modified from AdaptiveImageJacobian, region controller with adaptive or precise Js
    """
    # def __init__(self, fa: FrankaArm =None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
    def __init__(self, fa=None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None, update_mode=0) -> None:
        # n_k_per_dim => the number of rbfs in each dimension

        if fa is None:
            raise ValueError('FrankaArm handle is not provided!')
        else:
            self.fa = fa

        # dimensions declaration
        self.m = 6  # the dimension of Cartesian space configuration r
        self.n_k = n_k_per_dim ** 3  # the dimension of rbf function ??(r)
        # Js has three variables (x, y, z), 
        # and (??i, ??j, ??k) do not actually affect Js, 
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
                # raise ValueError('Target point x on the image plane should not be empty!')
                x = np.array([MyConstants.IMG_W/2,MyConstants.IMG_H/2])
            fx, fy = MyConstants.FX_HAT, MyConstants.FY_HAT 
            u0, v0 = MyConstants.U0, MyConstants.V0 
            fx, fy = MyConstants.FX_HAT, MyConstants.FY_HAT 
            u0, v0 = MyConstants.U0 , MyConstants.V0
            u, v = x[0] - u0, x[1] - v0
            z = 1
            """
                This Js seems wrong
            """
            # J_cam2img = np.array([[fx/z, 0,    -u/z, -u*v/fx,      (fx+u**2)/fx, -v], \
            #                       [0,    fy/z, -v/z, -(fy+v**2)/fy, u*v/fy,       u]])
            """
                This Js seems right
            """
            J_cam2img = np.array([[fx/z, 0,    -u/z, 0, 0, 0], \
                                  [0,    fy/z, -v/z, 0, 0, 0]])

            # my_jacobian_handler = ()
            # J_base2cam = my_jacobian_handler.calcJacobian(from_frame='panda_link0', to_frame='camera_link')
            # print(J_base2cam)

            R_c2b = np.array([[-0.99851048, -0.0126514,   0.05307315],
                              [-0.01185424,  0.99981255,  0.01530807],
                              [-0.05325687,  0.01465613, -0.99847329]])
            J_base2cam = np.block([[R_c2b,np.zeros((3,3))],[np.zeros((3,3)),R_c2b]])

            # print('J_base2cam',J_base2cam)
            """
            p_s_in_panda_EE = np.array([0.067, 0.08, -0.05])
            ee_pose_quat = fa.get_pose().quaternion[[1,2,3,0]]
            ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
            p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(3,1)
            p_s_cross = np.array([[0, -p_s[2], p_s[1]], \
                                [p_s[2], 0, -p_s[0]], \
                                [-p_s[1], p_s[0], 0]])
            J_p_cross = np.block([[np.eye(3), - p_s_cross],[np.zeros((3,3)),np.zeros((3,3))]])
            """

            p_s_in_panda_EE = np.array([0.067, 0.08, -0.05])  # panda_EE???marker???????????????
            ee_pose_quat = fa.get_pose().quaternion[[1,2,3,0]]
            ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
            p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(-1,)
            cross_mat = np.array([[0,        p_s[2], -p_s[1]],
                                  [-p_s[2],  0,       p_s[0]],
                                  [p_s[1],  -p_s[0],  0]])
            J_base2cam = np.block([[R_c2b, R_c2b], [np.zeros((3, 6))]])
            J_p_cross = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), cross_mat]])


            # rot_ee = fa.get_pose().rotation  # (rotation matrix of the end effector)
            # (r, p, y) = R.from_matrix(rot_ee).as_euler('XYZ', degrees=False)  # @TODO: intrinsic rotation, first 'X', second 'Y', third'Z', to be checked
            # J_baserpy2w = np.block([[np.eye(3), np.zeros((3, 3))], \
            #                         [np.zeros((3, 3)), np.array([[1, 0, math.sin(p)], \
            #                                                      [0, math.cos(r), -math.cos(p) * math.sin(r)], \
            #                                                      [0, math.sin(r), math.cos(p) * math.cos(r)]])]])
            self.Js_hat = J_cam2img @ J_base2cam @ J_p_cross  # Js_hat = J_base2img

            # Js initialization for adaptive control
            if update_mode !=0:
                # Js_hat_for_init = np.array([[-500,0,-500,-100,-100,100],[0,500,500,-100,-100,100]])
                # Js_hat_for_init = np.array([[-500,0,-500,-500,-500,500], [0,500,500,-500,-500,500]])
                # Js_hat_for_init = np.array([[-5000, 0, -1000, -100, -100, 100], \
                #                             [-300, 4500, 1000, -100, -100, 100]])
                # Js_hat_for_init = MyConstants.JS_HAT_FOR_INIT
                Js_hat_for_init = self.Js_hat * np.random.rand(2, 6) * 2
                self.Js_hat = Js_hat_for_init
                self.Js_hat = np.array([[-3.41610702e+02, -1.61563681e+01,  1.24743619e+02,  2.91767826e+00, -3.05454969e+02, 5.96228784e+01],
                                        [-3.09275739e+01, 4.49796283e+03, 1.98420993e+01, -3.79928865e+02, -4.89143663e-01, 2.38961637e+02]])
                print(Js_hat_for_init)

        """
            adaptive control with xyz coordinates
        """
        # cfg = {'n_dim':3,'n_k_per_dim':10,'sigma':1,'pos_restriction':np.array([[-0.2,0.4], [0.35,0.95], [-0.2,0.4]])}
        """
            adaptive control with xy coordinates
        """
        cfg = {'n_dim':2,'n_k_per_dim':50,'sigma':0.3/7,'pos_restriction':np.array([[0.05,0.35],[0.45,0.75]])}
        cfg = {'n_dim':2,'n_k_per_dim':3,'sigma':0.3/3,'pos_restriction':np.array([[0.05,0.35],[0.45,0.75]])}
        self.theta = RadialBF(cfg=cfg)
        self.theta.init_rbf_()

        if L is not None:
            if L.shape != (self.n_k, self.n_k):  # (1000, 1000)
                raise ValueError('Dimension of L should be ' + str((self.n_k, self.n_k)) + '!')
            self.L = L
        else:
            # self.L = np.eye(self.theta.n_k) * 10000
            self.L = np.ones((self.theta.n_k,self.theta.n_k)) * 1
            # raise ValueError('Matrix L should not be empty!')

        if W_hat is not None:
            if W_hat.shape != (2 * self.m, self.n_k):  # (12, 1000)
                raise ValueError('Dimension of W_hat should be ' + str((2 * self.m, self.n_k)) + '!')
            self.W_hat = W_hat
        else:
            self.W_hat = np.zeros((2*6, self.theta.n_k)) # initial all w are zeros
            # raise ValueError('Matrix W_hat should not be empty!')
        self.W_init_flag = False  # inf W_hat has been initialized, set the flag to True

        self.image_space_region = ImageSpaceRegion(b=np.array([MyConstants.IMG_W, MyConstants.IMG_H]))

        self.cartesian_space_region = CartesianSpaceRegion()
        self.cartesian_quat_space_region = CartesianQuatSpaceRegion()
        self.cartesian_space_region.set_r_c(MyConstants.CARTESIAN_CENTER)  # set by jyp | grasping pose above the second object with marker
        self.cartesian_space_region.set_c(MyConstants.CARTESIAN_SIDE_LENGTH)
        self.cartesian_space_region.set_Kc(MyConstants.CARTESIAN_KC)

        self.cartesian_quat_space_region.set_q_g(MyConstants.QUATERNION_QG)  # grasping pose on the right
        self.cartesian_quat_space_region.set_Ko(MyConstants.QUATERNION_KO)
        self.cartesian_quat_space_region.set_orient_range(MyConstants.QUATERNION_RANGE_COEFF)

        self.joint_space_region = JointSpaceRegion()
        # self.joint_space_region.add_region_multi(np.array([0, 0, 0, -3.08, 0, 0, 0]), 0.38, 0.58, np.array([0,0,0,1,0,0,0]), kq=20, kr=10, inner=True) # avoid the joint 4 entering [-2.9,-2.3] 

    def kesi_x(self, x):
        return self.image_space_region.kesi_x(x.reshape(1, -1))

    def kesi_r(self, r):
        return self.cartesian_space_region.kesi_r(r.reshape(1, -1))

    def kesi_rq(self, rq):
        return self.cartesian_quat_space_region.kesi_rq_omega(rq.reshape(-1,)) / 2

    def kesi_q(self, q):
        return self.joint_space_region.kesi_q(q.reshape(1, 7))

    def get_theta(self, r):
        r = r.reshape(-1,)[:self.theta.n_dim]
        rbf = self.theta.get_rbf_(r)
        # print('Dimension for vector theta: %d' % rbf.reshape(-1,).shape[0])
        return rbf

    def update_Js_with_ps(self, x, quat, z, update_mode=0):
        x = x.reshape(-1,)
        ee_pose_quat = quat[[1, 2, 3, 0]]
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        p_s_in_panda_EE = np.array([0.067, 0.08, -0.05])
        p_s = (ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)).reshape(-1,)
        Z = MyConstants.CAM_HEIGHT - z

        R_b2c = np.array([[-0.99851048, -0.0126514,   0.05307315],
                          [-0.01185424,  0.99981255,  0.01530807],
                          [-0.05325687,  0.01465613, -0.99847329]])
        Js = np.array([[MyConstants.FX_HAT / Z, 0,    - (x[0] - MyConstants.U0) / Z, 0, 0, 0], \
                        [0,    MyConstants.FY_HAT / Z, - (x[1] - MyConstants.V0) / Z, 0, 0, 0]])
        cross_mat = np.array([[0,        p_s[2], -p_s[1]],
                            [-p_s[2],  0,       p_s[0]],
                            [p_s[1],  -p_s[0],  0]])
        Jrot = np.block([[R_b2c, R_b2c], \
                        [np.zeros((3, 6))]])
        Jvel = np.block([[np.eye(3), np.zeros((3, 3))], \
                        [np.zeros((3, 3)), cross_mat]])

        if update_mode == 0:
            self.Js_hat = (Js @ Jrot @ Jvel).reshape(2, 6)
            return self.Js_hat
        else:
            return (Js @ Jrot @ Jvel).reshape(2, 6)

    def get_Js_hat(self):
        return self.Js_hat

    def get_u(self, J, d, r_t, r_o, q, x, with_vision=False, update_mode = 0):
        J_pinv = J.T @ np.linalg.pinv(J @ J.T)

        kesi_x = MyConstants.KESI_X_SCALE * self.kesi_x(x).reshape(-1, 1)  # (2, 1)

        kesi_r = self.kesi_r(r_t.reshape(1, 3))  # (1, 3)
        if self.cartesian_quat_space_region.fo(Quat(r_o)) <= 0:
            kesi_rq = np.zeros((1, 3))
        else:
            kesi_rq = self.cartesian_quat_space_region.kesi_rq_omega(r_o) / 5 # (1, 3)
        kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)
        self.kesi_rall = kesi_rall

        kesi_q = self.kesi_q(q).reshape(7, 1)  # (7, 1)

        # print("self.Js_hat.T @ kesi_x",self.Js_hat.T @ kesi_x)
        # print("kesi_rall",kesi_rall)
        value_to_be_compensated = 0  # initial value
        if with_vision:
            if update_mode==0:
                updated_Js_hat = self.update_Js_with_ps(x, r_o.reshape(-1,), r_t.reshape(-1,)[2])
                u = - J_pinv @ (updated_Js_hat.T @ kesi_x + kesi_rall + J @ kesi_q)  # normal version in paper
            elif update_mode==-1:
                u = - J_pinv @ (self.Js_hat.T @ kesi_x + kesi_rall + J @ kesi_q)
            elif update_mode==1:
                self.update(J, r_t, r_o, q, x)
                u = - J_pinv @ (self.Js_hat.T @ kesi_x + kesi_rall + J @ kesi_q)
            else:
                raise ValueError('update_mode could only be -1,0,1 !')

            # log the last terms in V_dot for debug
            if update_mode != 0:
                true_Js = self.update_Js_with_ps(x, r_o.reshape(-1,), r_t.reshape(-1,)[2], update_mode=update_mode).reshape(2, 6)
                Js_tilde = true_Js - self.Js_hat
                value_to_be_compensated = - 1e3 * (self.Js_hat.T @ kesi_x + kesi_rall + J @ kesi_q).T @ Js_tilde.T @ kesi_x
                # print('=====> value to be compensated: %.10f' % value_to_be_compensated)
                # print('kesi_r: ', kesi_rall.flatten())
                # print('kesi_q: ', kesi_q.flatten())
                # print('kesi_x: ', kesi_x.flatten())
                # print('dqd_x: ', - J_pinv @ self.Js_hat.T @ kesi_x)
        else:
            u = - J_pinv @ (kesi_rall + J @ kesi_q)

        # print('kesi_x', kesi_x)
        # print('kesi_rall', kesi_rall)
        # print('u_image_part', (- J_pinv @ (self.Js_hat.T @ kesi_x)).reshape(-1,))
        # print('u_cartesian_part', (- J_pinv @ kesi_rall).reshape(-1,))
        # print('u_joint_part', (- J_pinv @ (J_pinv.T @ kesi_q)).reshape(-1,))
        return u, float(value_to_be_compensated)

    def update(self, J=None, r_t=None, r_o=None, q=None, x=None, lock=None): # used when adaptive, if u are precise, don't use it
        # print('==================================')
        update_start_T = time.time()
        if self.fa is not None:
            r = self.fa.get_pose()  # get r: including translation and rotation matrix
            # split Cartesian translation and quaternion
            r_tran = r.translation
            r_quat = r.quaternion  # w, x, y, z
            # get the analytic jacobian (6*7)
            J = self.fa.get_jacobian(q)
        else:
            r_tran = r_t
            r_quat = r_o
            # J, q and x are passed through parameters
        update_get_data_T = time.time()
        # print('get data: %.5f' % (update_get_data_T - update_start_T))

        theta = self.get_theta(r_tran).reshape(-1, 1)  # get the neuron values theta(r) (1000*1)
        if not self.W_init_flag:
            for r_idx in range(self.W_hat.shape[0]):
                self.W_hat[r_idx, :] = (self.Js_hat.flatten()[r_idx] / np.sum(theta))
            self.W_init_flag = True

        update_get_theta_T = time.time()
        # print('get theta: %.5f' % (update_get_theta_T - update_get_data_T))

        
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        update_matinv_T = time.time()
        # print('mat inv: %.5f' % (update_matinv_T - update_get_theta_T))

        kesi_x = self.kesi_x(x).reshape(-1, 1)  # (2, 1)
        kesi_rt = self.kesi_r(r_tran).reshape(-1, 1)  # (3, 1)

        # print('kesi_rt/distance: ', kesi_rt.reshape(-1,)/(r_tran-MyConstants.CARTESIAN_CENTER)).reshape(-1) # yxj 0720

        kesi_rq = self.kesi_rq(r_quat)  # (1, 4) @ (4, 3) = (1, 3)
        kesi_rall = np.r_[kesi_rt, kesi_rq.reshape(3, 1)]  # (6, 1)
        kesi_q = self.kesi_q(q).reshape(-1, 1)  # (7, 1)

        kesi_x_prime = np.c_[kesi_x[0] * np.eye(6), kesi_x[1] * np.eye(6)]  # (6, 12)

        # update_calc_kesi_T = time.time()
        # print('calc kesi: %.5f' % (update_calc_kesi_T - update_matinv_T))

        # dW_hat = - self.L @ theta @ (self.Js_hat.T @ kesi_x + kesi_rall + J_pinv.T @ kesi_q).T  # (1000, 6)
        # dW_hat = dW_hat @ kesi_x_prime  # (1000, 12)
        dW_hat = calculate_dW_hat(self.L,theta,self.Js_hat,kesi_x,kesi_rall,J_pinv,kesi_q,kesi_x_prime)
        pdb.set_trace()
        """
        pool = Pool(processes=1)
        params = {'L': self.L, \
                  'theta': theta, \
                  'Js_hat': self.Js_hat, \
                  'kesi_x': kesi_x, \
                  'kesi_rall': kesi_rall, \
                  'J_pinv': J_pinv, \
                  'kesi_q': kesi_q, \
                  'kesi_x_prime': kesi_x_prime}
        matrix_mult_proc = pool.apply_async(MatrixMultiplication, args=(params,))
        pool.close()
        pool.join()
        dW_hat = matrix_mult_proc.get()
        try:
            dW_hat = dW_hat.reshape(self.theta.n_k, 12)
        except:
            exit(0)
        """
        update_Js_update_T = time.time()
        # print('Js update: %.5f' % (update_Js_update_T - update_calc_kesi_T))

        self.W_hat = self.W_hat + dW_hat.T

        # update J_s
        temp_Js_hat = self.W_hat @ theta  # (12, 1)
        self.Js_hat = np.c_[temp_Js_hat[:6], temp_Js_hat[6:]].T
        # Js_hat_candidate = np.c_[temp_Js_hat[:6], temp_Js_hat[6:]].T
        # if (Js_hat_candidate[0,0]<=self.Js_hat[0,0]+0.1 and Js_hat_candidate[1,1]<=self.Js_hat[1,1]+1):
        #     self.Js_hat = np.c_[temp_Js_hat[:6], temp_Js_hat[6:]].T  # (2, 6)
        # else:
        #     self.Js_hat[0,0] = self.Js_hat[0,0]-5
        # self.Js_hat[:,3:] = np.zeros((2,3))
        print((self.Js_hat.T @ kesi_x + kesi_rall + J_pinv.T @ kesi_q).T)

        # print('==================================')


# ===========================================================================
class FrankaInfoStruct(object):
    def __init__(self) -> None:
        self.x = np.array([0, 0])
        self.trans = np.array([0, 0, 0])
        self.quat = np.array([0, 0, 0, 0])

desired_position_bias = np.array([-170, -100])

class ImageDebug(object):
    def __init__(self, fa):
        # self.nh_ = rospy.init_node('image_debugging', anonymous=True)
        # self.img_sub = rospy.Subscriber('/aruco_simple/result', Image, callback=self.image_callback, queue_size=1)
        self.pixel1_sub = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, callback=self.pixel1_callback, queue_size=1)
        # self.pose_sub = rospy.Subscriber('/gazebo_sim/ee_pose', Float64MultiArray, self.pose_callback, queue_size=1)
        self.pixel2_sub = rospy.Subscriber('/aruco_simple/pixel2', PointStamped, callback=self.pixel2_callback, queue_size=1)
        self.img_pub = rospy.Publisher('/image_debug/result', Image, queue_size=1)
        self.pixel1 = None
        self.pixel1_queue = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        self.vel = np.array([0.0, 0.0])
        self.goal = None
        self.data_c = FrankaInfoStruct()
        self.pose_ready = False
        self.fa = fa

    def get_Js(self,data_c): # COPIED from image_debug.py, to get real Js! yxj 0712
        x = data_c.x.reshape(-1,)
        # ee_pose_quat = pose.quaternion[[1,2,3,0]]
        ee_pose_quat = data_c.quat[[1, 2, 3, 0]]
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        # p_s_in_panda_EE = np.array([0.058690, 0.067458, -0.053400])
        p_s_in_panda_EE = np.array([0.067, 0.08, -0.05])
        p_s = (ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)).reshape(-1,)

        # Z = MyConstants.CAM_HEIGHT - pose.trans.reshape(-1,)[2]
        Z = 1

        R_b2c = np.array([[-1, 0,  0],
                        [0,  1,  0],
                        [0,  0, -1]])
        Js = np.array([[MyConstants.FX_HAT / Z, 0,    - (x[0] - MyConstants.U0) / Z, 0, 0, 0], \
                        [0,    MyConstants.FY_HAT / Z, - (x[1] - MyConstants.V0) / Z, 0, 0, 0]])
        cross_mat = np.array([[0,        p_s[2], -p_s[1]],
                            [-p_s[2],  0,       p_s[0]],
                            [p_s[1],  -p_s[0],  0]])
        Jrot = np.block([[R_b2c, R_b2c], \
                        [np.zeros((3, 6))]])
        Jvel = np.block([[np.eye(3), np.zeros((3, 3))], \
                        [np.zeros((3, 3)), cross_mat]])

        return (Js @ Jrot @ Jvel).reshape(2, 6)

    def image_callback(self, data):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(data, 'bgr8')

        image_space_region = ImageSpaceRegion(b=np.array([MyConstants.IMG_W, MyConstants.IMG_H]))
        image_space_region.set_x_d(self.goal)
        image_space_region.set_Kv(np.array([2, 1]) / 10)

        if self.pixel1 is not None:
            kesi_x = image_space_region.kesi_x(self.pixel1.reshape(1, 2)).reshape(-1,)
            kesi_x = kesi_x / np.linalg.norm(kesi_x, ord=2)
            end_of_kesi_arrow = self.pixel1 - 150 * kesi_x
            cv2.arrowedLine(img, (round(self.pixel1[0]), round(self.pixel1[1])), 
                            (round(end_of_kesi_arrow[0]), round(end_of_kesi_arrow[1])), (0, 255, 0), \
                            thickness=4, line_type=cv2.LINE_4, shift=0, tipLength=0.2)

            end_of_vel_arrow = self.pixel1 + 125 * self.vel
            cv2.arrowedLine(img, (round(self.pixel1[0]), round(self.pixel1[1])), 
                            (round(end_of_vel_arrow[0]), round(end_of_vel_arrow[1])), (255, 0, 0), \
                            thickness=4, line_type=cv2.LINE_4, shift=0, tipLength=0.2)

            self.data_c.trans = self.fa.get_pose().translation
            self.data_c.quat = self.fa.get_pose().quaternion
            
            Js = self.get_Js(self.data_c)

            x = self.pixel1
            u, v = x[0] - MyConstants.U0, x[1] - MyConstants.V0
            # Z = MyConstants.CAM_HEIGHT - z
            Z = 1
            R_c2i = np.array([[MyConstants.FX_HAT / Z, 0, - u / Z],
                                [0, MyConstants.FY_HAT / Z, - v / Z]])

            V_b = - (Js.T @ kesi_x).reshape(-1,)[:3]
            V_b = V_b.reshape(3, 1)

            R_b2c = np.array([[-1, 0,  0],
                            [0,  1,  0],
                            [0,  0, -1]])
            V_c = (R_b2c @ V_b).reshape(3, 1)
            V_i = (R_c2i @ V_c).reshape(-1,)
            V_i = V_i / np.linalg.norm(V_i, ord=2)
            end_of_vel_arrow = self.pixel1 + 100 * V_i
            cv2.arrowedLine(img, (round(self.pixel1[0]), round(self.pixel1[1])), 
                            (round(end_of_vel_arrow[0]), round(end_of_vel_arrow[1])), (0, 0, 255), \
                            thickness=4, line_type=cv2.LINE_4, shift=0, tipLength=0.2)

            ratio = kesi_x.reshape(-1,) / V_i.reshape(-1,)
            cv2.putText(img, 'kesi/V = (%.5f, %.5f)' % (ratio[0], ratio[1]), (40, 100), \
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        if self.goal is not None:
            cv2.circle(img, (round(self.goal[0]), round(self.goal[1])), 8, color=(0, 0, 255), thickness=-1)

        self.img_pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))

    def pixel1_callback(self, data):
        x = np.array([data.point.x, data.point.y])
        self.pixel1 = x
        self.pixel1_queue.pop(0)
        self.pixel1_queue.append(x.tolist())
        pixel1_array = np.array(self.pixel1_queue)
        self.vel = (pixel1_array[-1, ...] - pixel1_array[0, ...]).reshape(-1,)
        self.vel = self.vel / np.linalg.norm(self.vel, ord=2)
        self.data_c.x = x

    def pixel2_callback(self, data):
        self.goal = np.array([data.point.x, data.point.y]) + desired_position_bias

    # def pose_callback(self, data):
    #     self.data_c.trans = np.array(data.data)[:3].reshape(3,)
    #     self.data_c.quat = np.array(data.data)[[6, 3, 4, 5]].reshape(4,)
    #     self.pose_ready = True

    def main(self):
        rospy.spin()


# 0702 yxj
def test_adaptive_region_control(fa:FrankaArm, update_mode=0):

    pre_traj = "./data/0724/my_debug_image_"+time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))+"_"+str(update_mode)+"/"

    desired_position_bias = np.array([-170, -90])

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

    # update_mode=0: true Js; -1:wrong Js no update; 1:wrong Js do update
    data_c = vision_collection()
    controller_adaptive = AdaptiveRegionController(fa,update_mode=update_mode)
    # img_debug = ImageDebug(fa)

    # nh_ = rospy.init_node('cartesian_joint_space_region_testbench', anonymous=True)
    sub_vision_1_ = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, data_c.vision_1_callback, queue_size=1)
    sub_vision_2_ = rospy.Subscriber('/aruco_simple/pixel2', PointStamped, data_c.vision_2_callback, queue_size=1)
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)  

    rate_ = rospy.Rate(100)

    print("wait for the object to be grasped! ")
    # init control scheme
    while 1:
        if data_c.is_data_without_vision_1_ready():
            target = data_c.x2
            target[0] = target[0]+desired_position_bias[0]
            target[1] = target[1]+desired_position_bias[1]
            controller_adaptive.image_space_region.set_x_d(target)
            controller_adaptive.image_space_region.set_Kv(np.array([4, 1]) / 20)
            print('vision region is set!')
            print(data_c.x2)
            print(target)
            break

    f_list, p_list, kesi_x_list, pixel_1_list, pixel_2_list, time_list, Js_list=[], [], [], [], [], [], []
    q_and_manipubility_list = np.zeros((0, 8))
    f_quat_list,p_quat_list,quat_list,kesi_rall_list,position_list = [],[],[],[],[]
    value_to_be_compensated_list = []

    # max_execution_time = 10.0
    max_execution_time = 30.0
    
    fa.goto_joints(joints=[2.33225779, 0.00270122, -1.43884068, -2.39329789, 1.47615966, 2.62341545, -1.21026707], ignore_virtual_walls=True, block=True)

    home_joints = fa.get_joints()
    fa.dynamic_joint_velocity(joints=home_joints,
                                joints_vel=np.zeros((7,)),
                                duration=max_execution_time,
                                buffer_time=10,
                                block=False)
    i=0

    change_the_space_region_param = False
    has_enter_vision_region_record = False
    has_enter_vision_region=0

    time_start = rospy.Time.now().to_time()
    
    # update control scheme
    while not rospy.is_shutdown():
        # print('==================================')
        iter_start_T = time.time()
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = fa.get_joints()
        J = fa.get_jacobian(q_and_m[0, :7])
        det = np.linalg.det(J @ J.T)
        q_and_m[0, 7] = math.sqrt(np.abs(det))
        pose = fa.get_pose()

        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)

        d = np.array([[0],[0],[0],[0],[0],[0]])
        pre_calc_data_T = time.time()
        # print('pre calc data: %.5f' % (pre_calc_data_T - iter_start_T))

        """
            region information
        """

        # print('In Cartesian Region: ', controller_adaptive.cartesian_space_region.in_region(pose.translation.reshape(-1,)))
        # print('In Quat Region: ', controller_adaptive.cartesian_quat_space_region.in_region(pose.quaternion.reshape(-1,)))
        # print('In joint Region: ', controller_adaptive.joint_space_region.in_region(q_and_m[0, :7]))
        # print('In vision Region: ', controller_adaptive.image_space_region.in_region(data_c.x1))
        # print('Current xyz: ', pose.translation.reshape(-1,))
        # print('Joint4: ', fa.get_joints()[3])

        """
            unless the marker attached to gripper is seen
            donnot update Js and exclude it from calculating dq_d_
        """
        if data_c.is_data_with_vision_1_ready():
            if not has_enter_vision_region_record:
                # controller_adaptive.cartesian_space_region.set_Kc(np.array([2e-4, 2e-4, 9e-6]).reshape(1, 3))
                # controller_adaptive.cartesian_quat_space_region.set_Ko(0)
                has_enter_vision_region = i
                has_enter_vision_region_record=True
            # print('In Vision Region: ', controller_adaptive.image_space_region.in_region(data_c.x1))
            # dq_d_ = pool.map(controller_adaptive.get_u, (J, d, pose.translation, pose.quaternion, q_and_m[0, :7], data_c.x1, True, update_mode))
            dq_d_, value_to_be_compensated = controller_adaptive.get_u(J, d, pose.translation, pose.quaternion, q_and_m[0, :7], data_c.x1, with_vision=True, update_mode=update_mode)
        else:
            dq_d_, value_to_be_compensated = controller_adaptive.get_u(J, d, pose.translation, pose.quaternion, q_and_m[0, :7], data_c.x1, with_vision=False, update_mode=update_mode)
        # print('Js: ', controller_adaptive.Js_hat)
        get_dq_T = time.time()
        # print('get dq: %.5f' % (get_dq_T - pre_calc_data_T))
        
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
        # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        dyn_msg_pub_T = time.time()
        # print('dyn msg pub: %.5f' % (dyn_msg_pub_T - get_dq_T))

        # logging
        time_list.append(time.time()-time_start)
        pixel_1_list.append(data_c.x1)
        pixel_2_list.append(data_c.x2)
        f_list.append(controller_adaptive.image_space_region.fv(data_c.x1))
        p_list.append(controller_adaptive.image_space_region.Pv(data_c.x1))
        kesi_x_list.append(controller_adaptive.image_space_region.kesi_x(data_c.x1).reshape((-1,)))
        Js_list.append(controller_adaptive.Js_hat.reshape(-1,))

        f_quat_list.append(controller_adaptive.cartesian_quat_space_region.fo(Quat(pose.quaternion)))
        p_quat_list.append(controller_adaptive.cartesian_quat_space_region.Po(Quat(pose.quaternion)))
        quat_list.append(pose.quaternion.tolist())
        kesi_rall_list.append(controller_adaptive.kesi_rall)
        position_list.append(pose.translation)
        value_to_be_compensated_list.append(value_to_be_compensated)

        logging_T = time.time()
        # print('logging: %.5f' % (logging_T - dyn_msg_pub_T))

        if time.time() - time_start >= max_execution_time or fa.get_joints().flatten()[3] <= -3.0:
            # terminate dynamic joint velocity control
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - time_start, should_terminate=True)
            ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
            pub.publish(ros_msg)
            break

        rate_.sleep()
        sleep_T = time.time()
        # print('sleep: %.5f' % (sleep_T - logging_T))
        # print('==================================')
  
    os.mkdir(pre_traj)
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
    plt.ylim([0,MyConstants.IMG_W])
    plt.title('vision position vs time')
    plt.savefig(pre_traj+'vision_position.jpg')

    plt.figure()
    pixel_1_list = np.array(pixel_1_list)
    pixel_1_pos_value_mask = (pixel_1_list[:, 0] >= 0) & (pixel_1_list[:, 1] >= 0)
    pixel_1_list = pixel_1_list[pixel_1_pos_value_mask, :].tolist()
    plt.plot(np.array(pixel_1_list)[:,0], np.array(pixel_1_list)[:,1],color='b',label = 'vision trajectory')
    plt.scatter(target[0], target[1],color='r',label = 'desired position')
    plt.xlim([0,MyConstants.IMG_W])
    plt.ylim([0,MyConstants.IMG_H])
    ax = plt.gca()
    ax.invert_yaxis()
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

    def plot_transparent_cube(ax,alpha_ = 0.1,x=10,y=20,z=30,dx=40,dy=50,dz=60):
        xx = np.linspace(x,x+dx,2)
        yy = np.linspace(y,y+dy,2)
        zz = np.linspace(z,z+dz,2)

        xx2,yy2 = np.meshgrid(xx,yy)
        ax.plot_surface(xx2,yy2,np.full_like(xx2,z),alpha=alpha_,color='r')
        ax.plot_surface(xx2,yy2,np.full_like(xx2,z+dz),alpha=alpha_,color='r')

        yy2,zz2 = np.meshgrid(yy,zz)
        ax.plot_surface(np.full_like(yy2,x),yy2,zz2,alpha=alpha_,color='r')
        ax.plot_surface(np.full_like(yy2,x+dx),yy2,zz2,alpha=alpha_,color='r')

        xx2,zz2 = np.meshgrid(xx,zz)
        ax.plot_surface(xx2,np.full_like(yy2,y),zz2,alpha=alpha_,color='r')
        ax.plot_surface(xx2,np.full_like(yy2,y+dy),zz2,alpha=alpha_,color='r')

    plt.figure()
    ax1 = plt.axes(projection='3d')
    position_array  = np.array(position_list)
    ax1.plot3D(position_array[:,0],position_array[:,1],position_array[:,2],label='traj')
    ax1.scatter(position_array[0,0],position_array[0,1],position_array[0,2],c='r',label='initial')
    ax1.scatter(position_array[has_enter_vision_region,0],position_array[has_enter_vision_region,1],position_array[has_enter_vision_region,2],label='enter')
    # ax1.scatter(position_array[200,0],position_array[200,1],position_array[200,2],c='b',label='t=5s')
    ax1.scatter(MyConstants.CARTESIAN_CENTER[0],MyConstants.CARTESIAN_CENTER[1],MyConstants.CARTESIAN_CENTER[2],c='g',label='goal region center')
    c = MyConstants.CARTESIAN_SIDE_LENGTH.reshape(-1,)
    plot_transparent_cube(ax1,0.1,MyConstants.CARTESIAN_CENTER[0]-c[0],MyConstants.CARTESIAN_CENTER[1]-c[1],MyConstants.CARTESIAN_CENTER[2]-c[2],
    2*c[0],2*c[1],2*c[2])
    plt.gca().set_box_aspect(( max(position_array[:,0])-min(position_array[:,0]), 
                            max(position_array[:,1])-min(position_array[:,1]), 
                            max(position_array[:,2])-min(position_array[:,2])))
    
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

    plt.figure()
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.plot(time_list,np.array(Js_list)[:,i],label = 'kesi')
        plt.scatter(time_list[has_enter_vision_region], np.array(Js_list)[has_enter_vision_region, i], c='r')
    plt.suptitle('Js')
    plt.savefig(pre_traj+'Js.jpg')

    plt.figure()
    plt.plot(time_list,q_and_manipubility_list[:,0:7],label = 'kesi')
    plt.title('q')
    plt.savefig(pre_traj+'q.jpg')

    plt.figure()
    plt.plot(time_list, value_to_be_compensated_list,label = 'value')
    plt.scatter(time_list[has_enter_vision_region], value_to_be_compensated_list[has_enter_vision_region], c='r')
    plt.title('value to be compensated')
    plt.savefig(pre_traj+'value_to_be_compensated.jpg')
    
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
            'position_list':position_list,\
            'Js_list':Js_list,\
            'value_to_be_compensated':value_to_be_compensated_list}
    with open(pre_traj + 'data.pkl', 'wb') as f:
        pickle.dump(info, f)
    plt.show()


if __name__ == '__main__':
    fa = FrankaArm()
    # test_joint_space_region_control(fa=fa)

    # proc = psutil.Process()
    # proc.cpu_affinity([0, 1, 2, 3, 4, 5, 6, 7])

    # lock = Lock()
    test_adaptive_region_control(fa=fa, update_mode=1)
