"""
    my_adaptive_control.py, 2022-05-01
    Copyright 2022 IRM Lab. All rights reserved.
"""
import collections
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
        self.joint_space_region.add_region_multi(qc=np.array([1.35924685,0.01409621,-0.03143654,-2.1,  0.24573268,  2.48413623,0.50480508]), \
                                                 qbound=0.12, qrbound=0.15, \
                                                 mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
                                                 kq=0.2, kr=0.1, \
                                                 inner=False, scale=np.ones((7, 1)))
        self.singularity_joint = np.array([1.19582476e+00, -1.79016522e-03, 3.56311106e-01, -2.51608346e+00, 4.75119009e-01, 3.31746127e+00, 5.93365287e-01])
        # self.joint_space_region.add_region_multi(qc=self.singularity_joint, \
        #                                          qbound=0.50, qrbound=0.45, \
        #                                          mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
        #                                          kq=1000000, kr=10000, \
        #                                          inner=True, scale=np.ones((7, 1)))# this is joint sigularity position: inner = True
        self.joint_space_region.add_region_multi(np.array([0, 0, 0, -2.6, 0, 0, 0]), 0.2, 0.4, np.array([0,0,0,1,0,0,0]), kq=1000, kr=100, inner=True) # avoid the joint 4 entering [-2.9,-2.3] 

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

# joint space test code of my adaptive control
def test_joint_space_region_control(fa):
    pre_traj = "./data/0716/my_joint_region/has_repulsion/"
    controller = JointOutputRegionControl(sim_or_real='real', fa=fa)

    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate_ = rospy.Rate(30)

    dq_d_list = []
    kesi_q_list = []
    dist_list = []
    t_list = []
    p_list = []
    q_and_manipubility_list = np.zeros((0, 8))

    max_execution_time = 10.0

    home_joints = fa.get_joints()
    fa.dynamic_joint_velocity(joints=home_joints,
                                joints_vel=np.zeros((7,)),
                                duration=max_execution_time,
                                buffer_time=10,
                                block=False)

    i=0

    time_start = time.time()

    while not rospy.is_shutdown():
        time_start_this_loop = time.time()
        q_and_m = np.zeros((1, 8))
        q_and_m[0, :7] = fa.get_joints()
        q_and_m[0, 7] = controller.calc_manipubility(fa.get_jacobian(q_and_m[0, :7]))
        dist = np.linalg.norm((controller.singularity_joint - q_and_m[0, :7]), ord=2)
        dist_list.append(dist)
        q_and_manipubility_list = np.concatenate((q_and_manipubility_list, q_and_m), axis=0)
        # print('time consumption1: ', time.time() - time_start_this_loop)
        # dq_d_, kesi_q = controller.get_dq_d_(q=data_c.q, d=np.zeros((7, 1)), J_sim=data_c.J, time_start_this_loop=time_start_this_loop)
        dq_d_, kesi_q = controller.get_dq_d_(q=q_and_m[0, :7], d=np.zeros((7, 1)), time_start_this_loop=time_start_this_loop)
        # print(dq_d_)
        dq_d_list.append(dq_d_.reshape(7,).tolist())
        kesi_q_list.append(kesi_q.reshape(7,).tolist())
        t_list.append(time_start_this_loop-time_start)
        p_list.append(controller.joint_space_region.Ps(q_and_m[0, :7]))
        # msg =JointVelocityCommand()
        # msg.dq_d = dq_d_.reshape(7,)
        # pub.publish(msg)

        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - time_start, 
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

        # print('time consumption4: ', time.time() - time_start_this_loop)

        if time.time() - time_start >= max_execution_time:
            # terminate dynamic joint velocity control
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - time_start, should_terminate=True)
            ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
            pub.publish(ros_msg)
            break

        rate_.sleep()

    np.save(pre_traj+'q_and_m.npy', q_and_manipubility_list)
    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.plot(t_list,dq_d_list)
    plt.legend(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
    ax.set_title("dq_d")
    ax = plt.subplot(2, 2, 2)
    plt.plot(t_list,kesi_q_list)
    ax.set_title("kesi_q")
    plt.legend(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
    ax = plt.subplot(2, 2, 3)
    plt.plot(t_list,dist_list)
    ax.set_title("distance")
    ax = plt.subplot(2, 2, 4)
    plt.plot(t_list,q_and_manipubility_list[:, 7])
    ax.set_title("manipubility")
    plt.savefig(pre_traj+'figure.jpg')

    plt.figure()
    plt.plot(t_list,p_list)
    plt.title('P')
    plt.savefig(pre_traj+'P.jpg')

    plt.figure()
    plt.plot(t_list,q_and_manipubility_list[:,0:7])
    plt.legend(['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
    plt.title('q')
    plt.savefig(pre_traj+'q.jpg')
    
    plt.show()
    info = {'dq_d': dq_d_list, \
            'kesi_q': kesi_q_list, \
            'dist': dist_list, \
            'q_and_manip': q_and_manipubility_list, \
            't_list': t_list}
    with open(pre_traj+'data.pkl', 'wb') as f:
        pickle.dump(info, f)

def plot_figures():
    import pickle
    with open('./data/0521/data.pkl', 'rb') as f:
        data1 = pickle.load(f)
    with open('./data/0521/data.pkl', 'rb') as f:
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

# yxj 20220715
    # jsr = JointSpaceRegion()
    # jsr.add_region_multi(qc=np.array([1.35924685,0.01409621,-0.03143654,-2.1,  0.24573268,  2.48413623,0.50480508]), \
    #                                             qbound=0.12, qrbound=0.15, \
    #                                             mask=np.array([1, 1, 1, 1, 1, 1, 1]), \
    #                                             kq=1, kr=0.01, \
    #                                             inner=False, scale=np.ones((7, 1)))
    # jsr.add_region_multi(np.array([0.5, 0.5, 0.5, -2.6, 0.5, 0.5, 0.5]), 0.2, 0.3, np.array([0,0,0,1,0,0,0]), kq=5000, kr=500, inner=True)

    # aaa = np.arange(-3,-2.2,0.02)
    # index = []
    # kesi_list = []
    # P_list = []
    # for i, joint in enumerate(aaa):
    #     index.append(aaa[i])
    #     q = np.array([0,0,0,joint,0,0,0])
    #     kesi_list.append(jsr.kesi_q(q)[0,3])
    #     P_list.append(jsr.Ps(q))
    # q = np.array([0,0,0,-2.4,0,0,0])

    # print([float('%.2f' %i) for i in index])
    # print([float('%.2f' %i) for i in kesi_list])
    # print([float('%.2f' %i) for i in P_list])

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot(aaa, P_list)
    # plt.title('P')

    # plt.figure()
    # plt.plot(aaa, kesi_list)
    # plt.title('kesi')

    # plt.show()

# yxj 20220716
    fa = FrankaArm()
    test_joint_space_region_control(fa=fa)
    # plot_figures()
    
