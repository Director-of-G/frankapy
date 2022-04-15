from gettext import translation
from turtle import position
import rospy
from geometry_msgs.msg import Point
from autolab_core import RigidTransform

import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import copy

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup


def start_franka_arm():
    fa = FrankaArm()
    rospy.loginfo('Moving the franka arm to home position!')
    fa.reset_joints(block=True)

    return fa
class haptic_subscrbe_handler(object):
    def __init__(self, franka_arm: FrankaArm) -> None:
        self.franka_arm = franka_arm
        self.franka_arm_pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.haptic_pos_sub = rospy.Subscriber(name='/haptic/position',
                                               data_class=Point,
                                               callback=self.haptic_position_callback)
        
        self.init_time = None
        self.position_ee_d = None
        self.command_idx = 0
        self.pose_ee_0 = RigidTransform()  # initial end effector pose (RigidTransform)

        self.get_franka_initial_state()

    def get_franka_initial_state(self):
        pose = self.franka_arm.get_pose()
        self.pose_ee_0 = pose
        self.init_time = rospy.Time.now().to_time()

    def haptic_position_callback(self, msg):
        self.position_ee_d = [msg.x, msg.y, msg.z]

    def main(self):
        start_time = time.time()
        while not rospy.is_shutdown():
            if time.time() - start_time >= 20:
                break

            timestamp = rospy.Time.now().to_time() - self.init_time

            print('pose_ee_0', self.pose_ee_0)

            traj_gen_proto_msg = PosePositionSensorMessage(
                id=self.command_idx, timestamp=timestamp, 
                position=self.pose_ee_0.translation, quaternion=R.from_matrix(self.pose_ee_0.rotation).as_quat()
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
            
            self.command_idx += 1

if __name__ == '__main__':    
    fa = start_franka_arm()
    
    haptic_sub = haptic_subscrbe_handler(fa)
    haptic_sub.main()

    rospy.spin()
