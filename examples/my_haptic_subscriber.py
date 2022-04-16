from gettext import translation
from re import T
from turtle import position, speed
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Int8MultiArray
from autolab_core import RigidTransform
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import time
import copy

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import convert_rigid_transform_to_array
import pickle as pkl

FILE_NAME = "/home/roboticslab/yxj/frankapy/data/0416"

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

    fa.goto_pose()

    return fa

def create_formated_skill_dict(joints, end_effector_positions, time_since_skill_started):
    skill_dict = dict(skill_description='GuideMode', skill_state_dict=dict())
    skill_dict['skill_state_dict']['q'] = np.array(joints)
    skill_dict['skill_state_dict']['O_T_EE'] = np.array(end_effector_positions)
    skill_dict['skill_state_dict']['time_since_skill_started'] = np.array(time_since_skill_started)

    # The key (0 here) usually represents the absolute time when the skill was started but
    formatted_dict = {0: skill_dict}
    return formatted_dict

class haptic_subscrbe_handler(object):
    def __init__(self, args: argparse.Namespace, franka_arm: FrankaArm) -> None:
        self.franka_arm = franka_arm
        self.franka_arm_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.haptic_pos_sub = rospy.Subscriber(name='/haptic/position',
                                               data_class=Point,
                                               callback=self.haptic_position_callback)
        self.haptic_but_sub = rospy.Subscriber(name='/haptic/button_state',
                                               data_class=Int8MultiArray,
                                               queue_size=1,
                                               callback=self.haptic_button_callback)

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

        self.get_franka_initial_state()
        self.initialize_controller()

    def get_franka_initial_state(self):
        pose = self.franka_arm.get_pose()
        self.pose_ee_0 = copy.deepcopy(pose)
        self.position_ee_0 = pose.translation
        self.orientation_ee_0 = pose.rotation
        self.T_ee_world = copy.deepcopy(self.franka_arm.get_pose())

    def initialize_controller(self):
        self.franka_arm.goto_pose(self.pose_ee_0, duration=self.record_time, dynamic=True, buffer_time=self.record_time,
                                  cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES)
        self.init_time = rospy.Time.now().to_time()

    def haptic_position_callback(self, msg):
        position_received = self.haptic_position_scale * np.array([msg.x, msg.y, msg.z])
        self.position_ee_d = position_received.tolist()
        assert len(self.position_ee_d) == 3 and np.all(np.abs(position_received) <= self.y_max_half_width)
        # print('[x, y, z] from Omega3: ', self.position_ee_d)

        """
            linearly mapping translation_y to rotation angle around the y-axis
            angle(y >= width_y) = angle_y
            angle(y <= -width_y) = -angle_y
        """
        rot_ang = (msg.y / 0.1) * self.max_rotation_y
        if rot_ang >= self.max_rotation_y:
            rot_ang = self.max_rotation_y
        elif rot_ang <= -self.max_rotation_y:
            rot_ang = -self.max_rotation_y

        T_ee_rot = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(np.deg2rad(rot_ang)),
            from_frame='franka_tool', to_frame='franka_tool'
        )

        self.rotation_ee_d = np.matmul(self.T_ee_world.rotation, T_ee_rot.rotation)

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

        rospy.loginfo("You can start moving omega 3 now!")
        start_time = time.time()
        while not rospy.is_shutdown():
            # print(self.command_idx)
            if time.time() - start_time >= self.record_time:
                break

            # record trajectory
            pose_array = convert_rigid_transform_to_array(fa.get_pose())
            end_effector_position.append(pose_array)
            joints.append(fa.get_joints())
            time_since_skill_started.append(time.time() - start_time)


            translation = copy.deepcopy(self.pose_ee_0).translation + np.array(self.position_ee_d)
            # pose_ee_delta = RigidTransform(rotation=np.ones((3, 3)), translation=self.position_ee_d, from_frame='franka_tool', to_frame='franka_tool')
            # pose_ee_current = self.pose_ee_0 * pose_ee_delta
            timestamp = rospy.Time.now().to_time() - self.init_time
            
            rot = R.from_matrix(copy.deepcopy(self.rotation_ee_d))
            quat = rot.as_quat()[[3, 0, 1, 2]]  # the quaternion is [x, y, z, w] in scipy and [w, x, y, z] in RigidTransform!

            # print('position: ', translation)
            # print('quaternion: ', quat)
            # print('T_ee_world', self.T_ee_world)
            # print('rotation_ee_d: ', self.rotation_ee_d)
            # print('quaternion_0: ', R.from_matrix(self.pose_ee_0.rotation).as_quat()[[3, 0, 1, 2]])
            # print('rot_desired', rot.as_matrix())
            # print('rot_ee_0', self.pose_ee_0.rotation)
            
            traj_gen_proto_msg = PosePositionSensorMessage(
                id=self.command_idx, timestamp=timestamp, 
                position=translation, quaternion=quat
            )
            fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                id=self.command_idx, timestamp=timestamp,
                translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
            self.franka_arm_pub.publish(ros_msg)

            self.command_idx += 1
            
            rate.sleep()


        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - self.init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        self.franka_arm_pub.publish(ros_msg)
        rospy.loginfo('Done')

        # save trajectory
        skill_dict = create_formated_skill_dict(joints, end_effector_position, time_since_skill_started)
        with open(FILE_NAME + '/traj.pkl', 'wb') as pkl_f:
            pkl.dump(skill_dict, pkl_f)
            print("Did save skill dict: {}".format(FILE_NAME + '/traj.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # constants
    parser.add_argument('--width_y', '-wy', type=str, default=0.45,
                        help='The maximum distance that end effector can move from home position along the y axis.')
    parser.add_argument('--angle_y', '-ay', type=str, default=70,
                        help='The maximum angle that end effector can rotate around the y axis.')
    parser.add_argument('--haptic_scale', '-hs', type=str, default=4,
                        help='The moving distance ratio of Franka end effector and haptic.')
    parser.add_argument('--record_time', '-rt', type=str, default=30,
                        help='Time length of the trajectory teaching.')
    parser.add_argument('--record_rate', '-rr', type=str, default=10,
                        help='Frequency of trajectory recording.')

    args = parser.parse_args()

    fa = start_franka_arm()

    rospy.sleep(5)
    
    haptic_sub = haptic_subscrbe_handler(args, fa)
    haptic_sub.main()

    rospy.spin()
