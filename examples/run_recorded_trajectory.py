import pickle as pkl
import numpy as np

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk, min_jerk_weight
import argparse

import rospy

if __name__ == "__main__":

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='franka_traj.pkl')
    args = parser.parse_args()

    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Generating Trajectory')

    # pose_traj = pkl.load(open('franka_traj.pkl','rb'))
    # pose_traj = pkl.load(open(args.file,'rb'))
    joints_traj = pkl.load(open('/home/roboticslab/yxj/frankapy/traj0408.pkl', 'rb'))
    joints_traj = joints_traj[0]["skill_state_dict"]['q']
    print(joints_traj.shape)
    

    T = 4.50
    dt = 0.01
    ts = np.arange(0, T, dt)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(1 / dt)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_joints(joints_traj[1], duration=T, use_impedance=False, dynamic=True, buffer_time=10, joint_impedances=FC.DEFAULT_JOINT_IMPEDANCES)
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        if i == len(joints_traj):
            break
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            joints=joints_traj[i]
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
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
    """
    

    
    joints_traj = pkl.load(open('/home/roboticslab/yxj/frankapy/traj0408.pkl', 'rb'))
    joints_traj = joints_traj[0]['skill_state_dict']['O_T_EE']
    joints_traj = joints_traj.reshape(len(joints_traj), 4, 4)
    pose_traj = []

    for i in range(len(joints_traj)):
        pose_traj.append(RigidTransform(rotation=joints_traj[i][:3, :3], 
                                        translation=joints_traj[i][:3, 3],
                                        from_frame='franka_tool',
                                        to_frame = 'world'))
    
    # exit()

    fa = FrankaArm()
    fa.reset_joints()

    rospy.loginfo('Generating Trajectory')

    # pose_traj = pkl.load(open('franka_traj.pkl','rb'))

    T = 10
    dt = 0.01
    ts = np.arange(0, T, dt)

    rospy.loginfo('Initializing Sensor Publisher')
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
    rate = rospy.Rate(110)

    rospy.loginfo('Publishing pose trajectory...')
    # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
    fa.goto_pose(pose_traj[1], duration=T, dynamic=True, buffer_time=10,
        cartesian_impedances=[600.0, 600.0, 600.0, 50.0, 50.0, 50.0]
    )
    init_time = rospy.Time.now().to_time()
    for i in range(2, len(ts)):
        timestamp = rospy.Time.now().to_time() - init_time
        traj_gen_proto_msg = PosePositionSensorMessage(
            id=i, timestamp=timestamp,
            position=pose_traj[i].translation, quaternion=pose_traj[i].quaternion
		)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            )

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
    
