"""
    my_utils.py, 2022-05-11
    Copyright 2022 IRM Lab. All rights reserved.
"""

import numpy as np
import math
import itertools

class Quat(object):  # quaternion class
    def __init__(self, value:np.ndarray=np.array([1, 0, 0, 0])) -> None:
        # in order of (w, x, y, z)
        if isinstance(value, list):
            self.quat = np.array(value)
        else:
            self.quat = value  # np.ndarray(4,)

    def w_(self):
        return self.quat[0]

    def x_(self):
        return self.quat[1]

    def y_(self):
        return self.quat[2]

    def z_(self):
        return self.quat[3]

    def v_(self):
        return self.quat[0]

    def u_(self):
        return self.quat[1:]

    def conjugate_(self):
        return Quat(np.r_[self.quat[0], -self.quat[1:]])

    def unit_(self):
        return Quat(self.quat / self.norm_(ord=2))

    def inverse_(self):
        return Quat(self.conjugate_().quat / (self.norm_(ord=2) ** 2))

    def product_(self, quat_):
        v1, u1 = self.v_(), self.u_()
        v2, u2 = quat_.v_(), quat_.u_()
        return Quat(np.r_[v1*v2 - np.dot(u1, u2), v1*u2 + v2*u1 + np.cross(u1, u2)])

    def norm_(self, ord=2):
        return np.linalg.norm(self.quat, ord=ord)
        
    def dq_(self, quat_):
        return self.product_(quat_.inverse_())

    def logarithm_(self, return_norm=True):
        if np.all(self.u_() == 0):
            if return_norm:
                return 0
            else:
                return np.zeros(3,)
        else:
            if return_norm:
                return math.acos(self.v_())
            else:
                return math.acos(self.v_()) * self.u_() / np.linalg.norm(self.u_(), ord=2)

    def dist_(self, quat_):
        d_quat = self.dq_(quat_)
        if np.all(d_quat == Quat(np.array([-1, 0, 0, 0]))):
            return 2 * math.pi
        else:
            return 2 * d_quat.logarithm_(return_norm=True)

    def angle_axis_(self):
        theta = 2 * math.acos(self.v_())
        axis = self.u_() / np.linalg.norm(self.u_(), ord=2)
        return theta * axis.reshape(1, -1)

    def __eq__(self, quat_) -> bool:
        return (self.w_() == quat_.w_()) and (self.x_() == quat_.x_()) and (self.y_() == quat_.y_()) and (self.z_() == quat_.z_())


class RadialBF(object):  # radial basis function(RBF) class
    def __init__(self, cfg:dict=None) -> None:
        self.n_dim = cfg['n_dim']  # number of dimensions (int), default=3
        self.n_k_per_dim = cfg['n_k_per_dim']  # number of n_k per dimension, default=10
        self.sigma = cfg['sigma']  # sigma_i for each dimension (1, n_dim) or use the uniform sigma (1,)
        self.pos_restriction = cfg['pos_restriction']  # Cartesian position restriction [p_low, p_high] in meters (3, 2)
        
        self.n_k = self.n_k_per_dim ** self.n_dim  # default=1000
        self.rbf_c_ = np.zeros((self.n_k, self.n_dim))  # default=(1000, 3)
        self.rbf_sigma2_ = np.zeros((self.n_k, 1))  # default=(1000, 1)

        self.init_rbf_()

    def init_rbf_(self):
        r_low, r_high = self.pos_restriction[:, 0], self.pos_restriction[:, 1]  # (3,), (3,)
        if not np.all(r_low <= r_high):
            raise ValueError('r_low <= r_high should be satisfied in Cartesian position restrictions')
        
        rbf_c = np.linspace(r_low, r_high, self.n_k_per_dim).T  # (3, n_k_per_dim)
        permute_idx = np.zeros_like(self.rbf_c_)  # default=(1000, 3)
        for col in range(permute_idx.shape[1]):
            permute_idx[:, col] = np.tile(np.repeat(np.arange(self.n_k_per_dim), self.n_k_per_dim ** (self.n_dim - col - 1)), self.n_k_per_dim ** col)

        for order, idx in enumerate(permute_idx):
            self.rbf_c_[:, order] = rbf_c[np.arange(self.n_dim), idx.reshape(-1,).tolist()]

        self.rbf_c_ = self.rbf_c_.reshape(self.n_k, self.n_dim)
        if len(self.sigma.shape) == 1:
            self.rbf_sigma2_ = np.array([self.sigma ** 2]).repeat(self.n_k).reshape(-1, 1)  # (n_k, 1)
        else:
            self.rbf_sigma2_ = np.array(self.sigma) ** 2

    def get_rbf_(self, r):
        r = r.reshape(1, 3)
        return np.exp(-(np.linalg.norm(r - self.rbf_c_, ord=2, axis=1) ** 2) / (2 * self.rbf_sigma2_))

        
