from frankapy.franka_arm import FrankaArm
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PointStamped

from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
print(sys.path)
from my_utils import Quat

from gazebo.my_gzb_adaptive_control import CartesianSpaceRegion, CartesianQuatSpaceRegion, ImageSpaceRegion
from torch.utils.tensorboard import SummaryWriter

from franka_example_controllers.msg import JointVelocityCommand
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm,SensorDataMessageType
from frankapy import FrankaConstants as FC

from geometry_msgs.msg import PointStamped

import pdb

class MyConstants(object):
    FX_HAT = 2337.218017578125
    FY_HAT = 2341.164794921875
    U0 = 746.3118044533257
    V0 = 564.2590475570069
    IMG_H = 1080
    IMG_W = 1440

def test_Image_Jacobian(fa):
    # Brief: define necessary classes
    class data_collection(object):  # to get Franka data from Gazebo
        def __init__(self) -> None:
            self.vision_1_ready = False
            self.vision_2_ready = False
            self.x1 = np.array([-1000,-1000])
            self.x2 = np.zeros((2,))

        def vision_1_callback(self, msg):
            self.x1 = np.array([msg.point.x,msg.point.y])
            if not self.vision_1_ready:
                self.vision_1_ready = True

        def vision_2_callback(self, msg):
            self.x2 = np.array([msg.point.x,msg.point.y])
            if not self.vision_2_ready:
                self.vision_2_ready = True

        def is_data_with_vision_ready(self):
            return self.vision_1_ready

    class traj_generator(object):
        def __init__(self, dir:np.ndarray=None, vel:np.ndarray=None, bound=200) -> None:
            self.dir = dir
            self.vel = vel
            self.bound = bound

            # FSM
            # 0 ==> stationary
            # 1 ==> positive direction
            # 2 ==> negative direction
            # 3 ==> transfer after positive direction
            # 4 ==> transfer after negative direction
            self.state = 0
            self.last_vel = np.zeros((2,))

        def in_region(self, x): # 判断是否在以图片center为中心的region内
            if (x[0] < self.bound or x[0] > MyConstants.IMG_W - self.bound) or \
               (x[1] < self.bound or x[1] > MyConstants.IMG_H - self.bound):
                return False

            else:
                return True

        def get_velocity(self, x): # 给region内来回跑的速度指令
            x = x.reshape(-1,)
            print('x: ', x)
            if self.state == 0:
                return np.zeros((2,))
            elif self.state == 1:
                if not self.in_region(x):
                    self.state = 3
                    print('Switching to state 3!')
                    return - self.vel * self.dir
                else:
                    return self.vel * self.dir
            elif self.state == 2:
                if not self.in_region(x):
                    self.state = 4
                    print('Switching to state 4!')
                    return self.vel * self.dir
                else:
                    return - self.vel * self.dir
            elif self.state == 3:
                if self.in_region(x):
                    self.state = 2
                    print('Switching to state 2!')
                return - self.vel * self.dir
            elif self.state == 4:
                if self.in_region(x):
                    self.state = 1
                    print('Switching to state 1!')
                return self.vel * self.dir
            else:
                return np.zeros((2,))

    def get_Js_hat(x, pose): # 
        ee_pose_quat = pose.quaternion
        ee_pose_mat = R.from_quat(ee_pose_quat).as_dcm()
        p_s_in_panda_EE = np.array([0.067, 0.08, -0.05])
        p_s = ee_pose_mat @ p_s_in_panda_EE.reshape(3, 1)

        z = 1.50 - pose.translation.reshape(-1,)[2]

        # R_b2c = np.array([[-1, 0,  0],
        #                   [0,  1,  0],
        #                   [0,  0, -1]])
        R_b2c = np.array([[-0.99851048, -0.0126514,   0.05307315],
                [-0.01185424,  0.99981255,  0.01530807],
                [-0.05325687,  0.01465613, -0.99847329]])
        Js = np.array([[MyConstants.FX_HAT/z, 0,    -MyConstants.U0/z, 0, 0, 0], \
                       [0,    MyConstants.FY_HAT/z, -MyConstants.V0/z, 0, 0, 0]])
        Js[0, 2], Js[1, 2] = -(x[0] - MyConstants.U0) / 0.5, -(x[1] - MyConstants.V0) / 0.5
        cross_mat = np.array([[0,        p_s[2,0], -p_s[1,0]],
                              [-p_s[2,0],  0,       p_s[0,0]],
                              [p_s[1,0],  -p_s[0,0],  0]])
        Jrot = np.block([[R_b2c, R_b2c], [np.zeros((3, 6))]])
        Jvel = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), cross_mat]])

        return (Js @ Jrot @ Jvel).reshape(2, 6)

    class velocity_calculator(object):
        def __init__(self) -> None:
            self.vels = np.zeros((3, 2)).tolist()
        
        def get_vel(self, x):
            x = x.reshape(-1,).tolist()
            self.vels.pop(0)
            self.vels.append(x)
            
            vels_array = np.array(self.vels)
            return (vels_array[2] - vels_array[0]) / (2 * (1 / 30))

    # nh_ = rospy.init_node('image_jacobian_testnode', anonymous=True)

    rate = rospy.Rate(30)
    data_c = data_collection()
    traj_g = traj_generator(dir=np.array([1, 0]), vel=0.01, bound=200)
    vel_c = velocity_calculator()
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    sub_vision_ = rospy.Subscriber('/aruco_simple/pixel1', PointStamped, data_c.vision_1_callback, queue_size=1)
    
    image_space_region = ImageSpaceRegion(b=np.array([1920, 1080]))
    cartesian_space_region = CartesianSpaceRegion()
    cartesian_quat_space_region = CartesianQuatSpaceRegion()

    # cartesian_space_region.set_r_c(np.array([-0.0711823860573642849, 0.48430624374805804, 0.669872105919327]))
    cartesian_space_region.set_r_c(np.array([0.23854596161538782, 0.5895254013581694, 0.3]))
    cartesian_space_region.set_c(np.array([0.05, 0.05, 0.05]).reshape(1, 3))
    cartesian_space_region.set_Kc(np.array([5e-5, 5e-5, 5e-5]).reshape(1, 3))

    cartesian_quat_space_region.set_q_g(np.array([-0.17492908847362298, 0.6884405719242297, 0.6818253503208791, 0.17479727175084528]))  # set by jyp | grasping pose above the second object with marker
    cartesian_quat_space_region.set_Ko(60)

    image_space_region.set_x_d(np.array([960.5, 540.5]))
    image_space_region.set_Kv(np.array([[0.2, 0.1]]))

    # writer = SummaryWriter(log_dir='./data/0708/' + time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())), flush_secs=5)
    writer = SummaryWriter(log_dir='./data/0709/my_test_image_jacobian/', flush_secs=5)

    # Step1: move the ArUco marker into FOV (with Cartesian region)
    max_execution_time = 25
    home_joints = fa.get_joints()
    fa.dynamic_joint_velocity(joints=home_joints,
                                joints_vel=np.zeros((7,)),
                                duration=max_execution_time,
                                buffer_time=10,
                                block=False)
    i=0
    time_start = rospy.Time.now().to_time()

    while not rospy.is_shutdown():
        q = fa.get_joints()
        pose = fa.get_pose()
        J, r_t, r_o = fa.get_jacobian(q), pose.translation, pose.quaternion

        J_pinv = J.T @ np.linalg.pinv(J @ J.T)
        kesi_r = cartesian_space_region.kesi_r(r_t.reshape(1, 3))  # (1, 3)

        if cartesian_quat_space_region.fo(Quat(r_o)) <= 0:
            kesi_rq = np.zeros((1, 3))
        else:
            kesi_rq = cartesian_quat_space_region.kesi_rq_omega(r_o) / 2 # (1, 3)
        kesi_rall = np.r_[kesi_r.T, kesi_rq.T]  # (6, 1)

        if data_c.is_data_with_vision_ready():
            x = data_c.x1
            pose = fa.get_pose()
            # Js_hat.T version
            kesi_x = image_space_region.kesi_x(x).reshape(2, 1)
            Js_hat = get_Js_hat(x, pose=pose).astype(np.float)  # (2, 6)
            dq_d_ = - J_pinv @ (Js_hat.T @ kesi_x + kesi_rall.reshape(6, 1))

            """
            # Js_hat_pinv version
            # remember to times kesi_x by 1e-6

            Js_hat_pinv = Js_hat.T @ np.linalg.inv(Js_hat @ Js_hat.T)  # (6, 2)
            dq_d_ = - J_pinv @ (Js_hat_pinv @ kesi_x + kesi_rall.reshape(6, 1))
            print('Js_hat_pinv: ', Js_hat_pinv)
            print('kesi x: ', kesi_x.reshape(-1,))
            print('vision output: ', (J_pinv @ Js_hat_pinv @ kesi_x).reshape(-1,))
            """
            print("is vision ready")
        else:
            dq_d_ = - J_pinv @ kesi_rall

        if np.linalg.norm(data_c.x1.reshape(-1,) - np.array([720.5, 540.5])) <= 50:
            time_now = rospy.Time.now().to_time() - time_start
            traj_gen_proto_msg = JointPositionVelocitySensorMessage(
                id=i, timestamp=time_now, 
                seg_run_time=max_execution_time,
                joints=home_joints,
                joint_vels=[0,0,0,0,0,0,0]
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
            )
            break
    
        time_now = rospy.Time.now().to_time() - time_start
        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=i, timestamp=time_now, 
            seg_run_time=max_execution_time,
            joints=home_joints,
            joint_vels=dq_d_.reshape(7,).tolist()
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        i += 1
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        print("state = 0")
        pub.publish(ros_msg)

        rate.sleep()

    # Step2: allow the ArUco marker center to have constant velocity in image space
    # cartesian_quat_space_region.set_Ko(0)
    # cartesian_space_region.set_Kc(np.array([0,0,0]).reshape(1, 3))
    traj_g.state = 1
    cnt = 0

    while not rospy.is_shutdown():
        pose = fa.get_pose()
        J = fa.get_jacobian(q)
        J_pinv = J.T @ np.linalg.pinv(J @ J.T)

        x = data_c.x1
        Js_hat = get_Js_hat(x, pose=pose)

        x_dot = traj_g.get_velocity(x).reshape(2, 1)
        dq_d_ = J_pinv @ (Js_hat.T @ x_dot)

        vel = vel_c.get_vel(x)  # list
        vel_x = {'desired': 10000*x_dot.reshape(-1,)[0],
                 'actual': vel[0]}
        vel_y = {'desired': 10000*x_dot.reshape(-1,)[1],
                 'actual': vel[1]}
        writer.add_scalars('vel_x', vel_x, global_step=cnt)
        writer.add_scalars('vel_y', vel_y, global_step=cnt)
        cnt += 1

        time_now = rospy.Time.now().to_time() - time_start
        traj_gen_proto_msg = JointPositionVelocitySensorMessage(
            id=i, timestamp=time_now, 
            seg_run_time=max_execution_time,
            joints=home_joints,
            joint_vels=dq_d_.reshape(7,).tolist()
        )
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
        )
        i += 1
        rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        # print(image_space_region.kesi_x(x).reshape(2, 1))
        print(Js_hat)
        print(Js_hat @ Js_hat.T)
        rate.sleep()

    writer.close()

def plot_image_region_vector_field():
    """
        1. Plot vector field in the image plane.
        2. For each position x in the image plane, project kesi_x
           to Cartesian space, then project the intermediate
           result to joint space.
        3. Obtain jacobian from frankapy (python package), project
           result of 2 to Cartesian space, as if it is executed on
           Franka Emika.
        4. Finally, project the resulting Cartesian velocity to
           image plane, as the vector at x.
    """

    # hyper-parameters
    IMG_W = 1920
    IMG_H = 1080
    NUM_W = 48
    NUM_H = 27

    W_LIST = np.linspace(0, IMG_W - 1, NUM_W).reshape(-1,)
    H_LIST = np.linspace(0, IMG_H - 1, NUM_H).reshape(-1,)
    AXIS_W, AXIS_H = np.meshgrid(W_LIST, H_LIST)

    z = 0.5  # Supposing z is fixed

    R_b2c = np.array([[-1, 0,  0],
                      [0,  1,  0],
                      [0,  0, -1]])  # Rotational matrix from {world} to {camera}
    
    def compute_R_c2i(x:np.ndarray):
        x = x.reshape(-1,)
        u, v = x[0] - MyConstants.U0, x[1] - MyConstants.V0
        Z = MyConstants.CAM_HEIGHT - z
        R_c2i = np.array([[MyConstants.FX_HAT / Z, 0, - u / Z],
                          [0, MyConstants.FY_HAT / Z, - v / Z]])
        
        return R_c2i

    data_c = FrankaInfoStruct()
    data_c.trans[2] = z
    image_space_region = ImageSpaceRegion(b=np.array([IMG_W, IMG_H]))
    image_space_region.set_x_d(np.array([960.5, 540.5]))
    image_space_region.set_Kv(np.array([2, 1]))

    vector_field = np.zeros((NUM_H, NUM_W, 2))
    kesi_x_field = np.zeros((NUM_H, NUM_W, 2))
    for v_idx, v in enumerate(H_LIST):
        for u_idx, u in enumerate(W_LIST):
            # get kesi_x
            x = np.array([u, v]).reshape(1, 2)
            data_c.x = x
            kesi_x = image_space_region.kesi_x(x).reshape(2, 1)
            kesi_x_field[v_idx, u_idx, :] = - kesi_x.reshape(-1,)

            # get Js
            Js = get_Js_hat(data_c)
            pdb.set_trace()

            # get unit vector V_i(x)
            """
                dq_d = - J_pinv @ (Js.T @ kesi_x)
                 V_b = J @ dq_d
                 V_b = - (Js.T @ kesi_x)
                 V_c = R_b2c @ V_b
                 V_i = R_c2i @ V_c
            """
            R_c2i = compute_R_c2i(x)
            V_b = - (Js.T @ kesi_x).reshape(-1,)[:3]
            V_b = V_b.reshape(3, 1)
            V_c = (R_b2c @ V_b).reshape(3, 1)
            V_i = (R_c2i @ V_c).reshape(-1,)
            vector_field[v_idx, u_idx, :] = 1e3 * V_i

    # plot figure
    pdb.set_trace()
    plt.quiver(AXIS_W, AXIS_H, vector_field[..., 0], vector_field[..., 1], color='blue', pivot='mid', width=0.001)
    plt.quiver(AXIS_W, AXIS_H, kesi_x_field[..., 0], kesi_x_field[..., 1], color='red', pivot='mid', width=0.001)
    plt.show()



if __name__ == '__main__':
    # test_zero_Jacobian()
    # test_plot_3D()
    fa = FrankaArm()
    test_Image_Jacobian(fa=fa)
