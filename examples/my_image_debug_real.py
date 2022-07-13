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

from my_adaptive_control import ImageSpaceRegion
# from gazebo.my_test_jyp import get_Js_hat, compute_R_c2i

import pdb

from frankapy import FrankaArm,SensorDataMessageType
from frankapy import FrankaConstants as FC

from scipy.spatial.transform import Rotation as R

IMG_W = 1440
IMG_H = 1080

class MyConstants(object):
    """
        @ Class: MyConstants
        @ Function: get all the constants in this file
    """
    FX_HAT = 2337.218017578125
    FY_HAT = 2341.164794921875
    U0 = 746.3118044533257
    V0 = 564.2590475570069
    CARTESIAN_CENTER = np.array([-0.0068108842682527, 0.611158320250102, 0.1342875493162069])

class FrankaInfoStruct(object):
    def __init__(self) -> None:
        self.x = np.array([0, 0])
        self.trans = np.array([0, 0, 0])
        self.quat = np.array([0, 0, 0, 0])

R_b2c = np.array([[-1, 0,  0],
                  [0,  1,  0],
                  [0,  0, -1]])

desired_position_bias = -np.array([200, 100])

class ImageDebug(object):
    def __init__(self, fa):
        # self.nh_ = rospy.init_node('image_debugging', anonymous=True)
        self.img_sub = rospy.Subscriber('/aruco_simple/result', Image, callback=self.image_callback, queue_size=1)
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
    
    def get_Js(self,data_c,pose): # COPIED from image_debug.py, to get real Js! yxj 0712
        x = data_c.x.reshape(-1,)
        ee_pose_quat = pose.quaternion[[1,2,3,0]]
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
        cross_mat = np.array([[0,        p_s[2,0], -p_s[1,0]],
                            [-p_s[2,0],  0,       p_s[0,0]],
                            [p_s[1,0],  -p_s[0,0],  0]])
        Jrot = np.block([[R_b2c, R_b2c], \
                        [np.zeros((3, 6))]])
        Jvel = np.block([[np.eye(3), np.zeros((3, 3))], \
                        [np.zeros((3, 3)), cross_mat]])

        return (Js @ Jrot @ Jvel).reshape(2, 6)

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


if __name__ == '__main__':
    fa = FrankaArm()
    debugger = ImageDebug(fa=fa)
    debugger.main()
