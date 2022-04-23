import rospy
from franka_example_controllers.msg import JointVelocityCommand
import numpy as np
import time

from matplotlib import pyplot as plt

def publish(pub):
    frequency = 40
    rate = rospy.Rate(frequency)
    max_velocity = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    delta_velocity = max_velocity / frequency / 4
    cnt = 0
    flag = 0
    joint_velocity_memory = []
    start_time = time.time()
    while not rospy.is_shutdown():
        joint_velocity = JointVelocityCommand()
        if flag == 0:
            dq_d = np.zeros((7, )) + cnt * delta_velocity
            cnt += 1
            if np.all(dq_d == max_velocity):
                flag = 1
                cnt = 0
        elif flag == 1:
            dq_d = max_velocity - cnt * delta_velocity
            cnt += 1
            if np.all(dq_d == np.zeros((7, ))):
                flag = 0
                cnt = 0
        joint_velocity_memory.append(dq_d.tolist())
        joint_velocity.dq_d = dq_d
        pub.publish(joint_velocity)
        print(time.time() - start_time)
        rate.sleep()
        if time.time() - start_time >= 16:
            break
    joint_velocity = JointVelocityCommand()
    joint_velocity.dq_d = np.zeros((7,))
    pub.publish(joint_velocity)
    plt.figure()
    plt.plot(joint_velocity_memory)
    plt.show()


if __name__ == '__main__':
    node = rospy.init_node('joint_velocity_publisher', anonymous=True)
    pub = rospy.Publisher('dyn_franka_joint_vel', JointVelocityCommand, queue_size=1)
    publish(pub)
