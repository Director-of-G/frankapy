from gettext import translation

import rospy
from geometry_msgs.msg import Point,PointStamped
from std_msgs.msg import Int8MultiArray
from autolab_core import RigidTransform
import argparse

import numpy as np
from pyquaternion import Quaternion
import time
import copy

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import convert_rigid_transform_to_array
import pickle as pkl

import math
from transformations import quaternion_from_matrix
from examples.my_hololens_reader import HololensPosition

# FILE_NAME = "/home/roboticslab/yxj/frankapy/data/0503"
FILE_NAME = "/home/roboticslab/yxj/frankapy/data/0721/my_haptic_subscriber_hololens/"

def start_franka_arm():
    fa = FrankaArm()
    rospy.loginfo('Moving the franka arm to home position 2!')
    # fa.reset_joints(block=True)
    # joints = fa.get_joints()
    # joints += np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.pi/2])
    joints = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, -np.pi / 4]
    fa.goto_joints(joints=joints, block=True)

    fa.open_gripper(block=True)
    time.sleep(3)
    fa.goto_gripper(width=0.02, 
                    grasp=True,
                    speed=0.04,
                    force=0.5,
                    block=True)

    return fa

def create_formated_skill_dict(joints, end_effector_positions, time_since_skill_started,log_d,desired_translation,desired_quat):
    skill_dict = dict(skill_description='GuideMode', skill_state_dict=dict())
    skill_dict['skill_state_dict']['q'] = np.array(joints)
    skill_dict['skill_state_dict']['O_T_EE'] = np.array(end_effector_positions)
    skill_dict['skill_state_dict']['time_since_skill_started'] = np.array(time_since_skill_started)
    skill_dict['skill_state_dict']['log_d'] = np.array(log_d)
    skill_dict['skill_state_dict']['desired_translation'] = np.array(desired_translation)
    skill_dict['skill_state_dict']['desired_quat'] = np.array(desired_quat)

    # The key (0 here) usually represents the absolute time when the skill was started but
    formatted_dict = {0: skill_dict}
    return formatted_dict

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

class haptic_subscrbe_handler(object):
    def __init__(self, args: argparse.Namespace, franka_arm: FrankaArm) -> None:
        self.franka_arm = franka_arm
        self.franka_arm_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

        self.position_ee_d = [0.0, 0.0, 0.0]  # desired end effector position in cartesion space
        self.rotation_ee_d = np.identity(3)
        self.orientation_ee_d = None  # desired end effector orientation in cartesion space
        
        self.position_ee_0 = None  # initial end effector position in cartesion space
        self.orientation_ee_0 = None  # initial end effector orientation in cartesion space
        self.pose_ee_0 = RigidTransform()  # initial end effector pose (RigidTransform)
        self.T_ee_world = RigidTransform()

        self.last_haptic_button = 0
        self.gripper_state = 'closed'  # 'opened', 'opening', 'closed', 'closing'
        self.is_grasped = True # check whether the object is grasped

        self.init_time = None
        self.command_idx = 0

        self.y_max_half_width = args.width_y  # the maximum distance the gripper moves from its origin position along the y axis
        self.max_rotation_y = args.angle_y  # the maximum angle the gripper can rotation around the y axis
        self.haptic_position_scale = args.haptic_scale  # if haptic device moves dy, then franka moves haptic_position_scale * dy
        self.record_time = args.record_time  # how long the trajectory teaching would last
        self.record_rate = args.record_rate  # trajectory recording rate
        self.save_traj = args.save_traj

        self.Kp = np.eye(6)#TODO
        self.Cd = 5*np.eye(7)#TODO

        self.get_franka_initial_state()
        self.initialize_controller()

        self.haptic_pos_sub = rospy.Subscriber(name='/haptic/position',
                                               data_class=Point,
                                               callback=self.haptic_position_callback)# this subscriber must be after getting initial state so that position_ee_0 is not none
        self.haptic_but_sub = rospy.Subscriber(name='/haptic/button_state',
                                               data_class=Int8MultiArray,
                                               queue_size=1,
                                               callback=self.haptic_button_callback)
    def get_franka_initial_state(self):
        pose = self.franka_arm.get_pose()
        self.pose_ee_0 = copy.deepcopy(pose)
        self.position_ee_0 = pose.translation
        self.orientation_ee_0 = pose.rotation
        self.T_ee_world = copy.deepcopy(self.franka_arm.get_pose())
        # print('self.pose_ee_0',self.pose_ee_0.matrix)
        self.r_d = np.reshape(np.concatenate((self.position_ee_0,quaternion_from_matrix(self.pose_ee_0.matrix)),axis=0),(7,1))

    def initialize_controller(self):
        self.home_joints = self.franka_arm.get_joints()
        self.franka_arm.dynamic_joint_velocity(joints=self.home_joints,
                                joints_vel=np.zeros((7,)),
                                duration=self.record_time,
                                buffer_time=10,
                                block=False)
        
        self.init_time = rospy.Time.now().to_time()

    def haptic_position_callback(self, msg):
        position_received = self.haptic_position_scale * np.array([msg.x, msg.y, msg.z])
        self.position_ee_d = position_received.tolist()
        assert len(self.position_ee_d) == 3

        rot_ang = (msg.y / 0.1) * self.max_rotation_y
        if rot_ang >= self.max_rotation_y:
            rot_ang = self.max_rotation_y
        elif rot_ang <= -self.max_rotation_y:
            rot_ang = -self.max_rotation_y

        T_ee_rot = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(np.deg2rad(rot_ang)),
            from_frame='franka_tool', to_frame='franka_tool'
        )

        # self.rotation_ee_d = np.matmul(self.T_ee_world.rotation, T_ee_rot.rotation)
        # rot = R.from_matrix(copy.deepcopy(self.rotation_ee_d))
        # quat = rot.as_quat()[[3, 0, 1, 2]]  # the quaternion is [x, y, z, w] in scipy and [w, x, y, z] in RigidTransform!
        # # print('quat',quat)
        self.rotation_ee_d = np.matmul(self.T_ee_world.matrix, T_ee_rot.matrix)
        try:
            self.r_d = np.concatenate((self.position_ee_d+\
                self.position_ee_0, 
            quaternion_from_matrix(self.rotation_ee_d)),axis=0)
            self.r_d = np.reshape(self.r_d, (7,1))
        except:
            print("self.position_ee_d",self.position_ee_d)
            print("self.position_ee_0",self.position_ee_0)
            print("self.rotation_ee_d",self.rotation_ee_d)
        
        # print('self.r_d',self.r_d)

    def haptic_button_callback(self, msg):
        # avoid shivering
        # print('haptic button | gripper_state | is_grasped ----------- ',self.last_haptic_button,msg.data[0],' | ', self.gripper_state,' | ',self.is_grasped)
        if self.gripper_state == "closing" and fa.get_gripper_width() <= 0.012:
            self.gripper_state = "closed"
            # here should add judgement for whether the object is actually grasped, now we assume is
            self.is_grasped = True
        if self.gripper_state == "opening" and fa.get_gripper_width() >= 0.073:
            self.gripper_state = "opened"
            self.is_grasped = False

        if (msg.data[0] == 1 and self.gripper_state == "opened" and not self.is_grasped):
            fa.goto_gripper(width=0.01,
                            grasp=True,
                            speed=0.04,
                            force=10,
                            block=False)
            self.gripper_state = "closing"
            self.is_grasped = True

        elif (msg.data[0] == 1 and self.gripper_state == "closed" and self.is_grasped):
            fa.goto_gripper(width=0.075,
                            speed=0.04,
                            grasp=False,
                            block=False)
            self.gripper_state = "opening"
            self.is_grasped = False

        self.last_haptic_button = msg.data[0]
        # time.sleep(1)
        

    def main(self):

        rate = rospy.Rate(self.record_rate)

        # record trajectory initialization
        end_effector_position = []
        joints = []
        time_since_skill_started = []
        log_d = []

        hololens_reader = HololensPosition()
        holo_sub = rospy.Subscriber("HoloLens_d", PointStamped, hololens_reader.callback2)

        rospy.loginfo("You can start moving omega 3 now!")
        start_time = time.time()

        desired_translation, desired_quat = [], []
        real_translation, real_quat = [], []
        send_v = []
        send_t = []

        i = 0
# ================================
        while not rospy.is_shutdown():
            # rospy.loginfo("start while-----")
            t = time.time()
            if t - start_time >= self.record_time:
                break

            # record trajectory
            pose_array = convert_rigid_transform_to_array(self.franka_arm.get_pose())
            end_effector_position.append(pose_array)
            joints.append(self.franka_arm.get_joints())
            time_since_skill_started.append(time.time() - start_time)
            

            translation = copy.deepcopy(self.pose_ee_0).translation + np.array(copy.deepcopy(self.position_ee_d))
            timestamp = rospy.Time.now().to_time() - self.init_time

            quat = quaternion_from_matrix(self.rotation_ee_d)  # the quaternion is [x, y, z, w] in scipy and [w, x, y, z] in RigidTransform!

            real_pose = self.franka_arm.get_pose()
            real_translation.append(real_pose.translation)
            real_quat.append(real_pose.quaternion)

            desired_translation.append(translation)
            desired_quat.append(quat)

            q_now = self.franka_arm.get_joints()
            r = pose_format(self.franka_arm.get_pose())
            J = self.franka_arm.get_jacobian(q_now)  # (6, 7)
            J_inv = np.linalg.pinv(J)  # (7, 6)
            J_pos = J[:3,:]
            J_pos_inv = np.linalg.pinv(J_pos)
            # N = np.eye(7) - np.dot(J_inv, J)
            N = np.eye(7) - np.dot(J_pos_inv, J_pos)

#===============================================
            joint4pos_my = hololens_reader.joint4pos
            J_hololens = self.franka_arm.get_jacobian_joint4(q_now)
            J_hololens_3d = J_hololens[:3,:]
            d1_3_dimension = np.dot(np.linalg.pinv(J_hololens_3d),joint4pos_my.reshape([3,1]))
            d1_3_dimension = np.asarray(d1_3_dimension).reshape(-1)
            d = np.concatenate((d1_3_dimension,np.array([0,0,0,0])),axis=0)
            d = np.reshape(d,(7,1))
            # d = np.array([[0],[0],[0],[0],[0],[0],[0]])
            
            log_d.append(d)
#===============================================

            # calculate the velocity ut
            # ut = - np.dot(J_inv, np.dot(self.Kp, error_format(r,self.r_d)))
            ut = - J_inv @ self.Kp @ error_format(r,self.r_d)
            # 计算un
            un = N @ np.linalg.inv(self.Cd) @ d

#===============================================
            # if t - start_time > 5 and t - start_time < 7:
            #     # d = -np.reshape(np.array([-0.2, -0.2, 0.2, 0.2, 0.2, 0.1, 0.1], float), (7, 1))
            #     joint4pos_my = np.array([0,0.2,0])
            #     J_hololens = self.franka_arm.get_jacobian_joint4(q_now)
            #     J_hololens_3d = J_hololens[:3,:]
            #     d1_3_dimension = np.dot(np.linalg.pinv(J_hololens_3d),joint4pos_my.reshape((3,1)))
            #     d1_3_dimension = np.asarray(d1_3_dimension).reshape(-1)
            #     d = np.concatenate((d1_3_dimension,np.array([0,0,0,0])),axis=0)
            #     d = np.reshape(d,(7,1))
            #     print('d',d)

            #     un = N @ np.linalg.inv(self.Cd) @ d
            #     print('un',un)
            #     # pass 
            # else:
            #     un = np.zeros((7,1))

            v = ut+un
            v = np.reshape(np.array(v), [-1,])
            
            v[v > 0.30] = 0.30
            v[v < -0.30] = -0.30

            send_v.append(v)
            send_t.append(t-start_time)

            traj_gen_proto_msg = JointPositionVelocitySensorMessage(
                id=i, timestamp=timestamp, 
                seg_run_time=30.0,
                joints=self.home_joints,
                joint_vels=v.tolist()
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
            )
            
            i += 1

            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.timestamp))
            self.franka_arm_pub.publish(ros_msg)

            rate.sleep()

            if self.gripper_state=="opened":
                break

        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - start_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
        self.franka_arm_pub.publish(ros_msg)

        # self.franka_arm.goto_gripper(0.02,grasp=True,speed=0.04,force=3,block=True)
        time.sleep(2)
        self.franka_arm.reset_joints(block=True)

        rospy.loginfo('Done')

        from matplotlib import pyplot as plt
        desired_translation_diff = np.array(desired_translation)[1:] - np.array(desired_translation)[:-1]
        desired_quat_diff = np.array(desired_quat)[1:] - np.array(desired_quat)[:-1]
        
                # save trajectory
        if self.save_traj:
            skill_dict = create_formated_skill_dict(joints, end_effector_position, time_since_skill_started,log_d,desired_translation,desired_quat)
            with open(FILE_NAME + 'traj.pkl', 'wb') as pkl_f:
                pkl.dump(skill_dict, pkl_f)
                print("Did save skill dict: {}".format(FILE_NAME + 'traj.pkl'))
        
        plt.figure(2)
        labels = ['x','y','z']
        lines = plt.plot(send_t,real_translation)
        plt.legend(lines, labels)
        plt.plot(send_t,desired_translation)
        plt.title("real translation and desired translation")

        plt.figure(3)
        plt.plot(real_quat)
        plt.plot(desired_quat)
        plt.title("real quaternion and desired quaternion")
 
        plt.figure(4)
        plt.plot(desired_translation_diff.tolist())
        plt.title("delta desired translation")
        
        plt.figure(5)
        plt.plot(desired_quat_diff.tolist())
        plt.title("delta desired quaternion")

        plt.figure(6)
        # labels = [1,2,3,4,5,6,7]
        # for y,label in zip(send_v,labels):
        #     plt.plot(y,label = label)
        plt.plot(send_t,send_v)
        plt.legend(['1','2','3','4','5','6','7'])
        plt.title("send_v")

        plt.show()
        
        print(desired_translation_diff.shape)
        print(desired_quat_diff.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # constants
    parser.add_argument('--width_y', '-wy', type=str, default=0.55,  # 0.45
                        help='The maximum distance that end effector can move from home position along the y axis.')
    parser.add_argument('--angle_y', '-ay', type=str, default=90,
                        help='The maximum angle that end effector can rotate around the y axis.')
    parser.add_argument('--haptic_scale', '-hs', type=str, default=6,  # 4
                        help='The moving distance ratio of Franka end effector and haptic.')
    parser.add_argument('--record_time', '-rt', type=str, default=50,
                        help='Time length of the trajectory teaching.')
    parser.add_argument('--record_rate', '-rr', type=str, default=10,
                        help='Frequency of trajectory recording.')
    parser.add_argument('--save_traj', '-st', action='store_true', default=True)

    args = parser.parse_args()

    fa = start_franka_arm()

    rospy.sleep(5)
    
    haptic_sub = haptic_subscrbe_handler(args, fa)
    haptic_sub.main()

    rospy.spin()
