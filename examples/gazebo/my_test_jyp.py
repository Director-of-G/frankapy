import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
import time

def test_Jacobian():
    class data_collection(object):
        def __init__(self) -> None:
            self.J = np.zeros((6, 7))
            self.J_ready = False

        def zero_jacobian_callback(self, msg):
            self.J = np.array(msg.data).reshape(6, 7)
            if not self.J_ready:
                self.J_ready = True

    data_c = data_collection()
        
    nh_ = rospy.init_node('cartesian_joint_space_region_testbench', anonymous=True)
    pub_ = rospy.Publisher('/gazebo_sim/joint_velocity_desired', Float64MultiArray, queue_size=10)
    sub_J_ = rospy.Subscriber('/gazebo_sim/zero_jacobian', Float64MultiArray, data_c.zero_jacobian_callback, queue_size=1)
    rate_ = rospy.Rate(100)
    
    while not data_c.J_ready:
        pass

    start_time = time.time()

    while not rospy.is_shutdown():
        body_twist = np.array([0, 0, -0.1, 0, 0, 0])
        J = data_c.J
        J_pinv = J.T @ np.linalg.inv(J @ J.T)
        dq_d = J_pinv @ body_twist.reshape(-1, 1)
        msg = Float64MultiArray()
        msg.data = dq_d.reshape(7,)
        pub_.publish(msg)                     
        rate_.sleep()

        if time.time() - start_time >= 5.0:
            msg = Float64MultiArray()
            msg.data = np.zeros(7,).reshape(7,)
            pub_.publish(msg)
            break

if __name__ == '__main__':
    test_Jacobian()
