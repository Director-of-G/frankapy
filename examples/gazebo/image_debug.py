"""
    Test code for image debugging node.
    Visualizing the following:
    - Camera image.
    - Direction of kesi_x.
    - Direction of desired marker velocity in the image plane.
    - Direction of actual marker velocity in the image plane.
"""

import cv2
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
import numpy as np

from my_gzb_adaptive_control import ImageSpaceRegion
from my_test_jyp import get_Js_hat, compute_R_c2i

import pdb

IMG_W = 1920
IMG_H = 1080

class FrankaInfoStruct(object):
    def __init__(self) -> None:
        self.x = np.array([0, 0])
        self.trans = np.array([0, 0, 0])
        self.quat = np.array([0, 0, 0, 0])

R_b2c = np.array([[-1, 0,  0],
                  [0,  1,  0],
                  [0,  0, -1]])

desired_position_bias = -np.array([240, 160])

class ImageDebug(object):
    def __init__(self):
        self.nh_ = rospy.init_node('image_debugging', anonymous=True)
        self.img_sub = rospy.Subscriber('/aruco_simple/result', Image, callback=self.image_callback, queue_size=1)
        self.pixel1_sub = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, callback=self.pixel1_callback, queue_size=1)
        self.pose_sub = rospy.Subscriber('/gazebo_sim/ee_pose', Float64MultiArray, self.pose_callback, queue_size=1)
        self.pixel2_sub = rospy.Subscriber('/aruco_simple/pixel2', PointStamped, callback=self.pixel2_callback, queue_size=1)
        self.img_pub = rospy.Publisher('/image_debug/result', Image)
        self.pixel1 = None
        self.pixel1_queue = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        self.vel = np.array([0.0, 0.0])
        self.goal = None
        self.data_c = FrankaInfoStruct()
        self.pose_ready = False

    def image_callback(self, data):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(data, 'bgr8')

        image_space_region = ImageSpaceRegion(b=np.array([IMG_W, IMG_H]))
        image_space_region.set_x_d(self.goal)
        image_space_region.set_Kv(np.array([2, 1]))

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

            if self.pose_ready:
                Js = get_Js_hat(self.data_c)
                R_c2i = compute_R_c2i(self.pixel1, self.data_c.trans[2])
                V_b = - (Js.T @ kesi_x).reshape(-1,)[:3]
                V_b = V_b.reshape(3, 1)
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

    def pose_callback(self, data):
        self.data_c.trans = np.array(data.data)[:3].reshape(3,)
        self.data_c.quat = np.array(data.data)[[6, 3, 4, 5]].reshape(4,)
        self.pose_ready = True

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    debugger = ImageDebug()
    debugger.main()
