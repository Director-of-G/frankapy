from cv2 import transform
from rospy import Publisher
import numpy as np

from geometry_msgs.msg import TransformStamped
import rospy
import tf


class TransformHandler(object):
    def __init__(self):
        self.nh_ = rospy.init_node(name='tf_handler', anonymous=True)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()
        self.marker_pose_sub = rospy.Subscriber('/aruco_single/transform', TransformStamped, self.callback_marker_pose, queue_size=1)

    def get_marker_ee_tf(self):
        (trans,rot) = self.tf_listener.lookupTransform('panda_EE', 'aruco_marker_frame', rospy.Time(0))
        print(trans)
        print(rot)

    def callback_marker_pose(self, msg):
        translation = [msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z]
        rotation = [msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w]
        self.tf_broadcaster.sendTransform(translation=np.array(translation), \
                                          rotation=np.array(rotation), \
                                          time=msg.header.stamp, \
                                          parent='camera_link', \
                                          child='aruco_marker_frame')


if __name__ == '__main__':
    tfhandler_ = TransformHandler()
    rate = rospy.Rate(5)
    rospy.sleep(3)
    while True:
        tfhandler_.get_marker_ee_tf()
        rate.sleep()
    rospy.spin()
