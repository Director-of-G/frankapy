import numpy as np

from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from franka_example_controllers.msg import JointVelocityCommand

from frankapy.utils import min_jerk

import rospy
import time


class JointVelsSubscriber(object):
    def __init__(self) -> None:
        self.joint_vels = np.zeros((7,))
        self.sub_ = rospy.Subscriber('/dyn_franka_joint_vel', JointVelocityCommand, self.callback, queue_size=1)

    def callback(self, msg):
        if not isinstance(msg.dq_d, np.ndarray):
            dq_d = np.array(msg.dq_d)
        else:
            dq_d = msg.dq_d
        self.joint_vels = dq_d

if __name__ == "__main__":
    fa = FrankaArm()

    joint_vels_sub_ = JointVelsSubscriber()
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

    fa.reset_joints(block=True)
    rospy.sleep(1)

    home_joints = fa.get_joints()

    max_execution_time = 30
    control_rate = 40
    rate = rospy.Rate(control_rate)

    fa.dynamic_joint_velocity(joints=home_joints,
                              joints_vel=np.zeros((7,)),
                              duration=max_execution_time,
                              buffer_time=10,
                              block=False)

    init_time = rospy.Time.now().to_time()
    start_time = time.time()
    i = 0
    while True:
        if time.time() - start_time > max_execution_time:
            break
        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            seg_run_time=30.0,
            joints=home_joints,
            joint_vels=joint_vels_sub_.joint_vels.tolist()
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        
        i += 1
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()

    # Stop the skill
    # Alternatively can call fa.stop_skill()
    term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
    pub.publish(ros_msg)

    rospy.loginfo('Done')
