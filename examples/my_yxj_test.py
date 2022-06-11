from cv2 import randn
from franka_example_controllers.msg import JointVelocityCommand
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm,SensorDataMessageType
from frankapy import FrankaConstants as FC

import numpy as np
import math

fa = FrankaArm()
# pose = fa.get_pose()
# print(pose)
# print(pose.position)
# print(pose.rotation)
# print(pose.quaternion)
log_m = []
log_j = []

for i in range(10000):
    joint = np.array([0.897782, 0.20624612, 0.53392278, -2.34249171, -0.20088835, 3.2333697, 0.97114771])+0.1*np.random.rand(7)
    J = fa.get_jacobian(joint)
    det = np.linalg.det(J @ J.T)
    m = math.sqrt(np.abs(det))
    log_m.append(m)
    log_j.append(joint)

print(log_m[np.argmin(log_m)])
print(log_j[np.argmin(log_m)])