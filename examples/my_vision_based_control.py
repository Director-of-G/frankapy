import imp
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from autolab_core import RigidTransform

import rospy
import numpy as np

from vision_pose_get import VisionPosition
# from hololens_reader import HololensPosition
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped

import time
from franka_example_controllers.msg import JointVelocityCommand
from geometry_msgs.msg import TransformStamped

from transformations import euler_from_matrix, quaternion_from_matrix

import copy
import math
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

import matplotlib.pyplot as plt

from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

import os
class CameraExtrinsics(object):
    """
        Class: CameraExtrinsics
        Function: record rigid transformation from camera frame to robot base frame
        @ translation: np.ndarray or list of size (3,)
        @ quaternion:  w, x, y, z
        @ rotation: rotation matrix of quaternion
    """
    def __init__(self, translation=None, quaternion=None, rotation=None) -> None:
        self.tranlation = translation
        if quaternion is not None:
            self.quaternion_set(quaternion)
        elif rotation is not None:
            self.rotation_set(rotation)
        self.rigid_transform = RigidTransform(from_frame='camera_frame', to_frame='world')

    def translation_set(self, translation):
        if isinstance(translation, np.ndarray):
            translation = translation.tolist()
        if not isinstance(translation):
            raise TypeError('parameter translation should be of type np.ndarray or list')
        if not len(translation)==3:
            raise ValueError('parameter translation should be of length 3')
        self.translation = translation
    
    def quaternion_set(self, quaternion):
        self.quaternion = quaternion
        self.rotation = R.from_quat((np.array(quaternion)[[1, 2, 3, 0]]).tolist()).as_matrix()
    
    def rotation_set(self, rotation):
        self.rotation = rotation
        self.quaternion = (R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]).tolist()

    def get_rigid_transform(self):
        self.rigid_transform.translation = self.tranlation
        self.rigid_transform.rotation = self.rotation

        return self.rigid_transform

class JointVelsSubscriber(object):
    """
        Class: JointVelsSubscriber
        Function: Subscribes ros topic '/dyn_franka_joint_vel' and takes down desired joint velocities
        Notes: Currently not used, because the publisher(vision based controller) and subscriber of joint velocities are in the same file
    """
    def __init__(self) -> None:
        self.joint_vels = np.zeros((7,))
        self.sub_ = rospy.Subscriber('/dyn_franka_joint_vel', JointVelocityCommand, self.callback, queue_size=1)

    def callback(self, msg):
        if not isinstance(msg.dq_d, np.ndarray):
            dq_d = np.array(msg.dq_d)
        else:
            dq_d = msg.dq_d
        self.joint_vels = dq_d

class MarkerPoseSubscriber(object):
    """
        Class: MarkerPoseSubscriber
        Function: Subscribes ros topic '/aruco_single/transform' and takes down the marker poses
        Notes: None
    """
    def __init__(self) -> None:
        self.sub_ = rospy.Subscriber('/aruco_single/transform', TransformStamped, self.marker_pose_callback, queue_size=1)
        self.marker_pose_camera_frame = RigidTransform(from_frame='aruco_marker', to_frame='camera_frame')
        self.marker_pose_robot_frame = RigidTransform()
        self.end_effector_robot_frame = FC.HOME_POSE
        self.end_effector_to_marker = RigidTransform(rotation=np.array([[0, 1, 0], \
                                                                        [1, 0, 0], \
                                                                        [0, 0, -1]]), \
                                                     from_frame='franka_tool', \
                                                     to_frame='aruco_marker')
        # the data was calibrated on 0430, after the camera position was higher
        self.camera_extrinsics = CameraExtrinsics(translation=[0.10854596161538782,0.5895254013581694,1.3748040178378818], \
                                                  quaternion=[0.026593157042844106,-0.006128866589368008,0.9995994598686369,0.00749405251881394])

        self.marker_pose_ready = False  # whether the marker pose has been stabalized or not

        self.desired_translation = []
        self.desired_orientation = []
        self.translation_memory = []
        self.orientation_memory = []
        self.translation_smooth_memory = []
        self.orientation_smooth_memory = []

    def calc_mean_quaternion(self, Quat):
        # deprecated
        """
        Quat = np.array(Quat)
        num_quat = Quat.shape[0]
        sum_Quat = np.zeros((4, 4)).astype(np.float)
        for i in range(num_quat):
            q = Quat[i, :].reshape(-1, 1)
            sum_Quat += np.matmul(q, q.T)
        sum_Quat /= num_quat
        """

        # In our task, quaternions to be considered are close to each other
        # Thus average is a reasonal solution
        sum_Quat = np.zeros((4,))
        Quat = np.array(Quat)
        num_quat = Quat.shape[0]
        for i in range(num_quat):
            quat = Quat[i, :]
            if np.dot(quat, Quat[0, :]) < 0:
                quat = -quat
            sum_Quat += quat
        sum_Quat /= num_quat

        return sum_Quat

    def marker_pose_callback(self, msg):
        if msg.header.frame_id == 'camera_link' and msg.child_frame_id == 'aruco_marker_frame':
            marker_translation = msg.transform.translation
            translation_received = [marker_translation.x, marker_translation.y, marker_translation.z]
            marker_quaternion = msg.transform.rotation
            quaternion_received = [marker_quaternion.w, marker_quaternion.x, marker_quaternion.y, marker_quaternion.z]
            rotation_received = R.from_quat((np.array(quaternion_received)[[1, 2, 3, 0]]).tolist()).as_matrix()
            received_transform = RigidTransform(translation=translation_received, \
                                                rotation=rotation_received, \
                                                from_frame='aruco_marker',
                                                to_frame='camera_frame')
            self.marker_pose_camera_frame = copy.deepcopy(received_transform)
            self.marker_pose_robot_frame = copy.deepcopy(self.camera_extrinsics.get_rigid_transform()) * copy.deepcopy(self.marker_pose_camera_frame)

            # print('marker pose in robot frame: ', self.marker_pose_robot_frame)
            
            desired_pose = self.marker_pose_robot_frame * self.end_effector_to_marker
            
            if len(self.desired_translation) > 40:
                self.desired_translation.pop(0)
            self.desired_translation.append(desired_pose.translation)
            if len(self.desired_orientation) > 40:
                self.desired_orientation.pop(0)
            self.desired_orientation.append(desired_pose.quaternion)

            average_translation = np.mean(np.array(self.desired_translation), axis=0)
            average_quaternion = self.calc_mean_quaternion(self.desired_orientation)
            self.end_effector_robot_frame = RigidTransform(rotation=R.from_quat(average_quaternion[[1, 2, 3, 0]]).as_matrix(), \
                                                           translation=average_translation)

            if not self.marker_pose_ready and len(self.desired_translation) == 40 and len(self.desired_orientation) == 40:
                self.marker_pose_ready = True
                                                
            self.translation_memory.append(desired_pose.translation)
            self.orientation_memory.append(desired_pose.quaternion)
            self.translation_smooth_memory.append(average_translation)
            self.orientation_smooth_memory.append(average_quaternion)

    def get_rigid_transform(self):
        return np.hstack((self.end_effector_robot_frame.translation, self.end_effector_robot_frame.quaternion)).reshape(7, 1)

def pose_format(pose_data):
    """
    return: 7x1
    """
    a = np.concatenate((pose_data.translation, pose_data.quaternion),axis=0)
    # print(a)
    return np.reshape(a,(7,1))

def error_format(r,r_d):
    """
    input: r,r_d 7x1 array; return:error=r-r_d 6x1 array
    """
    position_error_list = list(r[:3,0]-r_d[:3,0])
    orientation_d=Quaternion(r_d[3:,0])
    orientation = Quaternion(r[3:,0])
    # orientation_d = orientation_d.normalised
    # orientation = orientation.normalised
    # assert orientation_d.norm==1 and orientation.norm==1
    # print(orientation_d,' | ',orientation)

    if (np.dot(r_d[3:,0],r[3:,0]) < 0.0):
        orientation = -orientation

    error_quaternion = orientation * orientation_d.conjugate

    qw = error_quaternion[0]
    qx = error_quaternion[1]
    qy = error_quaternion[2]
    qz = error_quaternion[3]

    angle = 2 * math.acos(qw)
    x = qx / math.sqrt(1-qw*qw)
    y = qy / math.sqrt(1-qw*qw)
    z = qz / math.sqrt(1-qw*qw)

    error = position_error_list + [angle*x,angle*y,angle*z]
    return np.reshape(np.array(error),(6,1))


def main():
    dir = '/my_vision_based_control/0429/'
    current_path = os.path.dirname(__file__)

    print(current_path)
    
    fa = FrankaArm()
    
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    sub = MarkerPoseSubscriber()

    ros_freq = 30
    rate=rospy.Rate(ros_freq)

    vision_reader = VisionPosition()

    vision_sub1 = rospy.Subscriber("/aruco_simple/pixel1", PointStamped, vision_reader.callback1)
    vision_sub2 = rospy.Subscriber("/aruco_simple/pixel2", PointStamped, vision_reader.callback2)
    # hololens_reader = HololensPosition()
    # sub = rospy.Subscriber("UnityJointStatePublish", JointState, hololens_reader.callback)


    flag_initialized = 0

    # camera intrinsic parameters
    projection_matrix = np.reshape(np.array([2337.218017578125, 0, 746.3118044533257, 0,
                                             0, 2341.164794921875, 564.2590475570069, 0, 
                                             0, 0, 1, 0]), [3, 4])# according to ost0419.yaml
    fx = projection_matrix[0,0]
    fy = projection_matrix[1,1]
    u0 = projection_matrix[0,2]
    v0 = projection_matrix[1,2]

    # prepare for logging data
    qdot = np.zeros([7, 1],float)
    log_x = np.empty([2, 0],float)
    log_r = np.empty([7, 0],float)
    log_q = np.empty([7, 0],float)
    log_qdot = np.empty([7, 0],float)
    log_rdot = np.empty([6, 0],float)
    log_dqdot =  np.empty([7,0],float)
    v = [0, 0, 0, 0, 0, 0, 0]
    a = 50
    tt = 1

    x_d = np.reshape([720, 540], [2, 1])
    # r_d = np.reshape([0.5,0,0.5,0,1.0,0,0], [7, 1])#desired position and quaternion(wxyz)
    # r_d = np.reshape([0.13269275, 0.43067921, 0.28257956,-0.03379123,  0.88253785,  0.42634109, -0.19547999], [7, 1])
    # r_d = np.reshape([0.15238175, 0.57948289, 0.20484301, -0.0091628, 0.71664203, 0.69585671, 0.04608346], [7, 1])  # 码朝上
    # r_d = np.reshape([0.16252644, 0.58431606, 0.20523592, 0.06691753, 0.72849338, 0.68122359, -0.02745627], [7, 1])  # 码倾斜
    # r_d = np.reshape([3.06904962e-01, 1.11763435e-04, 4.86721445e-01, -1.40304221e-04, 9.99997553e-01, 2.32293991e-04, -7.84344880e-05], [7, 1])
    
    Kp = 0.5 * np.eye(2)#TODO
    Cd = np.eye(7)#TODO

    time.sleep(0.3)# wait for a short time otherwise q_last is empty
    q_last = fa.get_joints()
    x_last1 = vision_reader.pos1
    x_last2 = vision_reader.pos2
    # k_last = vision_reader.k_pos
    r_last = pose_format(fa.get_pose())
    t_start = time.time()
    t_last = t_start
    t_ready = t_start
    i = 0
    print("time begins at: ",t_start)
    # ================================================while begins
    while not rospy.is_shutdown():
        # rospy.loginfo("start while-----")

        t = time.time()
        q_now = fa.get_joints()
        qdot = fa.get_joint_velocities()
        
        if len(q_now)==0:
            print("can not get q !!!!!")
            continue
        x_now = vision_reader.pos1
        x_d = np.reshape(vision_reader.pos2, [2, 1])
        x_d = x_d-np.array([[340],[154]])
        x = x_now
        k_now = vision_reader.k_pos
        r_now = pose_format(fa.get_pose())
        r = r_now


        # 计算雅可比矩阵 J
        J = fa.get_marker_link_jacobian(q_now)  # (6, 7)
        J_inv = np.linalg.pinv(J)  # (7, 6)


        pose = fa.get_pose().matrix
        J_ori_inv = np.linalg.pinv(J[3:6, :]) # only compute orientation!!
        J_pos_inv = np.linalg.pinv(J[0:3, :])
        N = np.eye(7) - np.dot(J_inv, J)
        N_ori = np.eye(7) - np.dot(J_ori_inv, J[3:6, :])
        N_pos = np.eye(7) - np.dot(J_pos_inv, J[0:3, :])

        # 给指令
        if flag_initialized == 0:# 先回到初始位置
            # fa.reset_joints(block=True)
            # print('Resetting Franka to home joints!')
            # time.sleep(1)
            flag_initialized = 1

        elif flag_initialized == 1:
            home_joints = fa.get_joints()

            max_execution_time = 8

            fa.dynamic_joint_velocity(joints=home_joints,
                                      joints_vel=np.zeros((7,)),
                                      duration=max_execution_time,
                                      buffer_time=10,
                                      block=False)
            
            flag_initialized = 2
            print("initialization is done! the time is:", t - t_start)
            t_ready = t
            # fa.run_guide_mode(duration=20,block=False)

            RR = np.array([[-1, 0, 0], \
                           [0,  1, 0], \
                           [0, 0, -1]]) # 相机相对于base的旋转矩阵,要转置成base相对于相机才对
            RR = RR.T
            x = np.reshape(x, [2, 1])
            # theta_k0 = np.array([[f * RR[0, 0]], \
            #                      [f * RR[0, 1]], \
            #                      [f * RR[0, 2]], \
            #                      [f * RR[1, 0]], \
            #                      [f * RR[1, 1]], \
            #                      [f * RR[1, 2]], \
            #                      [RR[2, 0]],     \
            #                      [RR[2, 1]],     \
            #                      [RR[2, 2]],     \
            #                      [u00 * RR[2, 0]], \
            #                      [u00 * RR[2, 1]], \
            #                      [u00 * RR[2, 2]], \
            #                      [v00 * RR[2, 0]], \
            #                      [v00 * RR[2, 1]], \
            #                      [v00 * RR[2, 2]]])
            # theta_k = theta_k0

        elif flag_initialized == 2 and t - t_ready < max_execution_time: 
            # 计算视觉矩阵Js
            u = x[0] - u0
            v = x[1] - v0
            z = 1 # =========================
            Js = np.array([[fx/z, 0, -u/z], \
                           [0, fy/z, -v/z]])
            Js = np.dot(Js, RR)
            Js_inv = np.linalg.pinv(Js)

            # Js_hat = np.array([[theta_k[0,0]-theta_k[6,0]*x[0,0]+theta_k[9,0], theta_k[1,0]-theta_k[7,0]*x[0,0]+theta_k[10,0], theta_k[2,0]-theta_k[8,0]*x[0,0]+theta_k[11,0]],\
            #                                                 [theta_k[3,0]-theta_k[6,0]*x[1,0]+theta_k[12,0], theta_k[4,0]-theta_k[7,0]*x[1,0]+theta_k[13,0], theta_k[5,0]-theta_k[8,0]*x[1,0]+theta_k[14,0]]])

            # 末端位置 = r_now
            

            # 计算末端速度
            rdot = np.dot(J,np.reshape(qdot, (7,1)))

            # 更新目标位置
            # if sub.marker_pose_ready:
            #     r_d = sub.get_rigid_transform()

            # 计算ut
            x = np.reshape(x,[2,1])
            ut = -np.dot( J_pos_inv, np.dot( Js_inv , np.dot(Kp, (x-x_d) ) ) )

            # # 计算un
            # if t - t_ready > 15 and t - t_ready < 16:
            #     d = np.reshape(np.array([-0.2, -0.2, 0.2, 0.2, 0.2, 0.1, 0.1], float), [7, 1])
            #     un = -np.dot(N_pos, np.dot(np.linalg.inv(Cd), d)) 
            # else:
            #     un = np.zeros([7,1])
            
            v = ut 
            v = np.reshape(np.array(v), [-1,])

            # print(v," | ",(x-x_d).tolist())
            
            v[v > 0.10] = 0.10
            v[v < -0.10] = -0.10
            
            log_x = np.concatenate((log_x,np.reshape(x,(2,1))),axis=1)
            log_r = np.concatenate((log_r,np.reshape(r, (7,1))), axis=1)
            log_q = np.concatenate((log_q,np.reshape(q_now, (7,1))), axis=1)
            log_qdot = np.concatenate((log_qdot,np.reshape(qdot, (7,1))), axis=1)
            log_dqdot = np.concatenate((log_dqdot,np.reshape(v, (7,1))), axis=1)
            log_rdot = np.concatenate((log_rdot,np.reshape(rdot, (6,1))), axis=1)

            traj_gen_proto_msg = JointPositionVelocitySensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - t_ready, 
                seg_run_time=30.0,
                joints=home_joints,
                joint_vels=v.tolist()
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
            )
            
            i += 1
            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
            pub.publish(ros_msg)

        elif flag_initialized == 2 and t - t_ready > max_execution_time:
            # terminate dynamic joint velocity control
            term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - t_ready, should_terminate=True)
            ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
            pub.publish(ros_msg)
            fa.goto_gripper(0.02,grasp=True,speed=0.04,force=3,block=True)
            time.sleep(2)
            fa.reset_joints(block=True)
            flag_initialized = 3
        else:
            break

        q_last = q_now
        # k_last = k_now
        r_last = r_now
        t_last = t

        rate.sleep()# Sleeps for any leftover time in a cycle. Calculated from the last time sleep, reset, or the constructor was called. 这里就是ros_freq Hz
    # ===============================================while ends
    # print(time.time()-t_start)

    # print(np.shape(log_qdot))
    # log_x_array = np.array(log_x)

    np.save(current_path+ dir+'log_r.npy',log_r)
    np.save(current_path+ dir+'log_q.npy',log_q)
    np.save(current_path+dir+'log_qdot.npy',log_qdot)
    np.save(current_path+dir+'log_rdot.npy',log_rdot)
    np.save(current_path+dir+'log_dqdot.npy',log_dqdot)

    # task space velocity==============================================
    plt.figure(figsize=(30,20))
    for j in range(6):
        ax = plt.subplot(3, 2, j+1)
        ax.set_title('task space velocity %d' % (j+1),fontsize=20)
        plt.xlabel('time (s)')
        if j<3:
            plt.ylabel('velocity (m/s)')
        else:
            plt.ylabel('angular velocity (rad/s)')

        plt.plot(np.linspace(0,np.shape(log_rdot)[1]/ros_freq,np.shape(log_rdot)[1]),np.reshape(np.array(log_rdot[j,:]),[-1,]) ,label = 'actual veloc')
        plt.legend()
    plt.savefig(current_path+ dir+'log_rdot.jpg')

    # # vision space position===============================================
    plt.figure()
    plt.plot(log_x[0,:], log_x[1,:],label = 'actual')
    plt.scatter(x_d[0],x_d[1],label = 'target', c='r')
    plt.legend()
    plt.title('vision space trajectory')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.savefig(current_path+ dir+'log_x.jpg')

    # # vision space position verse time======================================
    fig = plt.figure(figsize=(20,8))
    plt.plot(np.linspace(0,np.shape(log_rdot)[1]/ros_freq,np.shape(log_rdot)[1]), log_x[0,:]-x_d[0],label = 'x')
    plt.plot(np.linspace(0,np.shape(log_rdot)[1]/ros_freq,np.shape(log_rdot)[1]), log_x[1,:]-x_d[1],label = 'y')
    plt.legend()
    # plt.title('vision space error')
    plt.xlabel('time (s)')
    plt.ylabel('error (pixel)')
    # plt.savefig(current_path+ dir+'log_x_t.jpg',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.savefig(current_path+ dir+'log_x_t.jpg')


    # joint space velocity=================================================
    plt.figure(figsize=(30,20))
    for j in range(7):
        ax = plt.subplot(4, 2, j+1)
        ax.set_title('joint space velocity %d' % (j+1),fontsize=20)
        plt.xlabel('time (s)')
        plt.ylabel('velocity (rad/s)')

        plt.plot(np.linspace(0,np.shape(log_qdot)[1]/ros_freq,np.shape(log_qdot)[1]),np.reshape(np.array(log_qdot[j,:]),[-1,]) ,label='actual joint velocity')
        plt.plot(np.linspace(0,np.shape(log_dqdot)[1]/ros_freq,np.shape(log_dqdot)[1]),log_dqdot[j,:].reshape(-1,), label = 'command joint velocity')
        plt.legend()
    plt.show()
    plt.savefig(current_path+ dir+'log_qdot.jpg')

def test():
    sub = MarkerPoseSubscriber()
    print('has subscribed!')
    start_time = time.time()
    while not rospy.is_shutdown():
        if time.time() - start_time > 15:
            break
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(sub.translation_memory)
    plt.subplot(1, 2, 2)
    plt.plot(sub.translation_smooth_memory)
    plt.show()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(sub.orientation_memory)
    plt.subplot(1, 2, 2)
    plt.plot(sub.orientation_smooth_memory)
    plt.show()

if __name__ == '__main__':
    main()

    # nh_ = rospy.init_node('vision_based_control_node', anonymous=True)
    # test()
    # rospy.spin()
