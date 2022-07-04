# """
#     tf currently should run with python2
# """
# import rospy
# import tf
# import pdb

# from scipy.spatial.transform import Rotation as R

# def tf_handler_server(listener):
#     # try:
#     listener.waitForTransform('/world', '/panda_EE', rospy.Time(0), rospy.Duration(0.1))
#     (trans,rot) = listener.lookupTransform('/world', '/panda_EE', rospy.Time(0))
#     rot = R.from_quat(rot)
#     pdb.set_trace()
#     rot = rot.as_matrix()
#     pdb.set_trace()
#     service = rospy.Service('look_up_trans')
#     # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
#     #     pdb.set_trace()
#     #     pass

# if __name__ == '__main__':
#     rospy.init_node('tf_handler_node', anonymous=True)
#     tf_listener = tf.TransformListener()
#     tf_handler_server(tf_listener)
