"""
    my_adaptive_control.py, 2022-05-01
    Copyright 2022 IRM Lab. All rights reserved.
"""
from frankapy.franka_arm import FrankaArm
import rospy
import tf

from scipy.spatial.transform import Rotation as R
import numpy as np
import math

from my_utils import Quat, RadialBF
import pdb

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
        return 0.5 * self.Kv * (1 - math.min(0, fv) ** 2)
    def kesi_x(self, x):  # x => (1, 2)
        partial_fv = 2 * (x - self.x_d) / (self.b ** 2)
        partial_fv = partial_fv.reshape(1, -1)
        return - self.Kv * min(0, self.fv(x)) * partial_fv

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
        fc = ((r - r_c) / c) ** 2 - 1
        return fc

    def in_region(self, r):
        fc = self.fc(r)
        return (fc <= 0).reshape(1, -1)

    def Pc(self, r):
        fc = self.fc(r)
        return np.sum(0.5 * self.Kc * np.max(0, fc) ** 2)

    def kesi_r(self, r, with_rot=False):
        if with_rot:  # deprecated
            r, r_c, c = r.reshape(6, 1), self.r_c.reshape(6, 1), self.c.reshape(6, 1)
            roll, yaw = r[3], r[5]
            if abs(roll - r_c[3]) > np.pi:
                roll = 2 * r_c[3] - roll
            if abs(yaw - r_c[5]) > np.pi:
                yaw = 2 * r_c[5] - yaw
            r[3], r[5] = roll, yaw
        partial_fc = 2 * (r - r_c) / (c ** 2)
        partial_fc = partial_fc.reshape(1, -1)
        return self.Kc * np.max(0, self.fc(r)) * partial_fc

class CartesianQuatSpaceRegion(object):
    def __init__(self, q_g:Quat=None, Ko=1) -> None:
        self.q_g = q_g  # Quat
        self.q_diff = Quat()  # Quat
        self.Ko = Ko  # Quat

    def set_q_g(self, q_g: Quat):
        self.q_g = q_g  # Quat

    def set_Ko(self, Ko):
        self.Ko = Ko  # float
    
    def fo(self, q:Quat, return_diff=False):
        self.q_diff = q.dq_(self.q_g)
        if return_diff:
            return self.Ko * self.q_diff.logarithm_(return_norm=True) - 1, self.q_diff
        else:
            return self.Ko * self.q_diff.logarithm_(return_norm=True) - 1

    def in_region(self, q:Quat):
        fo = self.fo(q)
        return (fo <= 0)

    def Po(self, q:Quat):
        fo = self.fo(q)

        return 0.5 * (max(0, fo)) ** 2

    def get_Jrot(self, q:Quat):
        r_o = q.angle_axis_()
        norm_r_o = np.linalg.norm(r_o, ord=2)
        J_rot = np.zeros((4, 3))
        J_rot[0, :] = - r_o * math.sin(norm_r_o / 2) / (2 * norm_r_o)
        J_rot_matBC = [[[r_o[0]**2, r_o[1]**2 + r_o[2]**2], [r_o[0]*r_o[1], -r_o[0]*r_o[1]], [r_o[0]*r_o[2], -r_o[0]*r_o[2]]], \
                       [[r_o[1]*r_o[0], -r_o[1]*r_o[0]], [r_o[1]**2, r_o[2]**2 + r_o[0]**2], [r_o[1]*r_o[2], -r_o[1]*r_o[2]]], \
                       [[r_o[2]*r_o[0], -r_o[2]*r_o[0]], [r_o[2]*r_o[1], -r_o[2]*r_o[1]], [r_o[2]**2, r_o[0]**2 + r_o[1]**2]]]
        J_rot_matBC = np.dot(np.array(J_rot_matBC), np.array([math.acos(norm_r_o/2)/(2*norm_r_o**2), math.sin(norm_r_o/2)/(norm_r_o**3)]))


    def kesi_rq(self, q:Quat):
        fo, q_diff = self.fo(q, return_diff=True)
        q_o, q_g = q.quat, self.q_g.quat
        partial_v_q = np.array([q_g[0] + q_o[0] * np.sum(q_g[1, 2, 3]/q_o[1, 2, 3]), \
                                -q_g[1] + q_o[1] * np.sum(q_g[[0, 2, 3]]/q_o[[0, 2, 3]]), \
                                -q_g[2] + q_o[2] * np.sum(q_g[[0, 1, 3]]/q_o[[0, 1, 3]]), \
                                -q_g[3] + q_o[3] * np.sum(q_g[[0, 1, 2]]/q_o[[0, 1, 2]])])
        u = q_diff[1:]
        norm_u = np.linalg.norm(u, ord=2)

        partial_P_q = (-(self.Ko / norm_u) * max(0, fo) * partial_v_q * (norm_u > 0)).reshape(1, -1)

        return (-(self.Ko / norm_u) * max(0, fo) * partial_v_q * (norm_u > 0)).reshape(1, -1)

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
        partial_f_q = np.zeros((1, 7))
        q = q.reshape(1, 7)
        q_scale_single = (q[0, self.single_mask.astype(np.int32)] * self.single_scale)  # (1, n_single)
        for i, idx in enumerate(self.single_mask[0]):
            partial_f_q[0, idx] += 2 * (q_scale_single[0, i] - self.single_qc[0, i]) * self.single_inout[0, i]
        
        q = q.reshape(7, 1)
        q_scale_multi = q * self.multi_scale  # (7, n_multi)
        partial_f_q += np.sum(2 * (q_scale_multi - self.multi_qc) * self.multi_inout * self.multi_mask, axis=1).reshape(1, 7)

        fq_single, fqr_single, fq_multi, fqr_multi = self.fq(q)

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
    def __init__(self, fa: FrankaArm =None, n_k_per_dim=10, Js=None, x=None, L=None, W_hat=None, theta_cfg:dict=None) -> None:
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
    def __init__(self, fa: FrankaArm =None) -> None:
        self.joint_space_region = JointSpaceRegion()
        if fa is None:
            raise ValueError('FrankaArm handle is not provided!')
        self.fa = fa
        self.cd = np.eye(7)

        self.joint_space_region = JointSpaceRegion()

    def init_joint_region(self):
        self.joint_space_region.add_region_multi(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 0.08, 0.1, np.array([1, 1, 1, 1, 1, 1, 1]), kq=5000, kr=10, inner=True)

    def get_dq_d_(self, q:np.ndarray, d:np.ndarray):
        q, d = q.reshape(7,), d.reshape(7, 1)
        J = self.fa.get_jacobian(q)  # get the analytic jacobian (6*7)
        J_pinv = J.T @ np.linalg.inv(J @ J.T)  # get the pseudo inverse of J (7*6)
        N = np.eye(7) - J_pinv @ J  # get the zero space matrix (7*7)
        kesi_q = self.joint_space_region.kesi_q(q)

        dq_d = - J_pinv @ (J_pinv.T @ kesi_q) + N @ np.linalg.inv(self.cd) @ d  # (7, 1)

        return dq_d


if __name__ == '__main__':
    jsr = JointSpaceRegion()
    jsr.add_region_multi(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 0.08, 0.1, np.array([1, 1, 1, 1, 1, 1, 1]), kq=5000, kr=10, inner=True)
    
    q_test = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    fq_s, fqr_s, fq_m, fqr_m = jsr.fq(q_test)
    print(fq_s, fqr_s, fq_m, fqr_m)
    fq_sin, fqr_sin = jsr.in_region(q_test)
    print(fq_sin, fqr_sin)
    Ps = jsr.Ps(q_test)
    print(Ps)
    kesi_q = jsr.kesi_q(q_test)
    print(kesi_q)

    q_test_list = []
    qd_list = []
    ps_list = []
    for qd in np.linspace(0.5, 0.75, 400):
        q_test = np.array([0.5, 0.5, qd, qd, 0.5, 0.5, 0.5])
        ps = jsr.Ps(q_test)
        kesi_q = jsr.kesi_q(q_test)
        q_test_list.append(kesi_q.squeeze())
        qd_list.append(qd)
        ps_list.append(ps)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(qd_list, ps_list)
    plt.ylim(0, 0.005)
    plt.show()
    
