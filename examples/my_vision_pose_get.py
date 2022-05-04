#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Vector3Stamped, TransformStamped

from autolab_core import RigidTransform

class VisionPosition():
    def __init__(self, name = "vision_info_subscriber"):
        self.name = name
        self.pos1 = []
        self.pos2 = [918, 790]
        self.actual_z = 0.0
        self.k_pos = 0
        self.ee_pose = RigidTransform(from_frame='franka_tool', to_frame='panda_link0')

    def Init_node(self):
        rospy.init_node(self.name)
        sub1 = rospy.Subscriber("/aruco_simple/pixel1", PointStamped, self.callback1)
        sub2 = rospy.Subscriber("/aruco_simple/pixel2", PointStamped, self.callback2)
        # actual_z_sub = rospy.Subscriber("/aruco_single/position", Vector3Stamped, self.callback_actual_z)
        # marker_pose_sub = rospy.Subscriber("/aruco_single/transform", TransformStamped, self.callback_transform_marker_to_ee_frame)
        # ee_pose_sub = rospy.Subscriber("/franka_info/ee_pose", TransformStamped, self.callback_ee_pose)

    def callback1(self, msg):
        # print([msg.point.x,msg.point.y])
        self.pos1 = list([msg.point.x,msg.point.y])
        # self.k_pos=msg.header.seq

    def callback2(self, msg):
        # print([msg.point.x,msg.point.y])
        self.pos2 = list([msg.point.x,msg.point.y])
        # self.k_pos=msg.header.seq

    def callback_actual_z(self,msg):
        self.actual_z = msg.vector.z
        print(self.actual_z)

    def callback_ee_pose(self, msg):
        self.ee_pose.translation = msg.transform.translation
        rot_mat = R.from_quat(msg.transform.rotation).as_matrix()
        self.ee_pose.rotation = rot_mat

    def callback_transform_marker_to_ee_frame(self, msg):
        # deprecated
        T_mark_cam = RigidTransform()
        T_mark_cam.translation = np.asarray(msg.transform.translation)
        rot_mat = R.from_quat(msg.rotation).as_matrix()
        T_mark_cam.quaternion = rot_mat
        T_mark_cam.from_frame = 'aruco_marker_frame'
        T_mark_cam.to_frame = 'camera_link'

        T_cam_base = RigidTransform()
        T_cam_base.translation = np.array([0.12814754119886385, 0.6123173, 0.7802863754115138])
        rot_max = R.from_quat(np.array([-0.00028621403739803597, 0.9998618117382161, -0.0057620385068332175, 0.01559084415106623])).as_matrix()
        T_cam_base.rotation = rot_mat
        T_cam_base.from_frame = 'camera_link'
        T_cam_base.to_frame = 'panda_link0'

        self.ee_pose * T_cam_base * T_mark_cam


if __name__ == "__main__":
    vision_info_reader = VisionPosition()
    vision_info_reader.Init_node()
    while not rospy.is_shutdown():
        # print(vision_info_reader.pos)
        pass
    