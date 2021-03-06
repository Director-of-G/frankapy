import imp
from frankapy import FrankaArm,SensorDataMessageType
from frankapy import FrankaConstants as FC
from autolab_core import RigidTransform

import rospy
import numpy as np

# from vision_pose_get import VisionPosition
# from hololens_reader import HololensPosition
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped

import time
from franka_example_controllers.msg import JointVelocityCommand

from transformations import euler_from_matrix, quaternion_from_matrix

import math
from pyquaternion import Quaternion

import matplotlib.pyplot as plt

from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionVelocitySensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

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



def pose_format(pose_data):
    """
    return: 7x1
    """
    a = np.concatenate((pose_data.translation, pose_data.quaternion),axis=0)
    # print(a)
    return np.reshape(a,(7,1))

def error_format(r,r_d):
    """
    input: r,r_d 7x1 array; return:error=r-r_d 6x1 array
    """
    position_error_list = list(r[:3,0]-r_d[:3,0])
    orientation_d=Quaternion(r_d[3:,0])
    orientation = Quaternion(r[3:,0])
    # orientation_d = orientation_d.normalised
    # orientation = orientation.normalised
    # assert orientation_d.norm==1 and orientation.norm==1
    # print(orientation_d,' | ',orientation)

    if (np.dot(r_d[3:,0],r[3:,0]) < 0.0):
        orientation = -orientation

    error_quaternion = orientation * orientation_d.conjugate

    qw = error_quaternion[0]
    qx = error_quaternion[1]
    qy = error_quaternion[2]
    qz = error_quaternion[3]

    angle = 2 * math.acos(qw)
    x = qx / math.sqrt(1-qw*qw)
    y = qy / math.sqrt(1-qw*qw)
    z = qz / math.sqrt(1-qw*qw)

    error = position_error_list + [angle*x,angle*y,angle*z]
    return np.reshape(np.array(error),(6,1))



def main():
    fa = FrankaArm()
    
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

    ros_freq = 30
    rate=rospy.Rate(ros_freq)

    # vel_cmd_pub = rospy.Publisher("/dyn_franka_joint_vel",JointVelocityCommand,queue_size=1)

    # vision_reader = VisionPosition()
    # vision_sub = rospy.Subscriber("/aruco_single/pixel", PointStamped, vision_reader.callback)
    # hololens_reader = HololensPosition()
    # sub = rospy.Subscriber("UnityJointStatePublish", JointState, hololens_reader.callback)

    flag_initialized = 0

    ## camera intrinsic parameters
    # projection_matrix = np.reshape(np.array([2340.00415, 0., 753.47694, 0., \
    #                                          0., 2342.75146, 551.78795, 0., \
    #                                          0., 0., 1., 0.]), [3, 4])
    # fx = projection_matrix[0,0]
    # fy = projection_matrix[1,1]
    # u0 = projection_matrix[0,2]
    # v0 = projection_matrix[1,2]

    # prepare for logging data
    qdot = np.zeros([7, 1],float)
    log_r = []
    log_q = np.empty([7, 0],float)
    log_qdot = np.empty([7, 0],float)
    log_rdot = np.empty([6, 0],float)
    log_dqdot =  np.empty([7,0],float)
    v = [0, 0, 0, 0, 0, 0, 0]
    a = 50
    tt = 1

    # dx = np.reshape([720, 540], [2, 1])
    # r_d = np.reshape([0.5,0,0.5,0,1.0,0,0], [7, 1])#desired position and quaternion(wxyz)
    r_d = np.reshape([0.13269275, 0.43067921, 0.28257956,-0.03379123,  0.88253785,  0.42634109, -0.19547999], [7, 1])
    
    Kp = 0.2 * np.eye(6)#TODO
    Cd = np.eye(7)#TODO

    time.sleep(0.3)# wait for a short time otherwise q_last is empty
    q_last = fa.get_joints()
    # x_last = vision_reader.pos
    # k_last = vision_reader.k_pos
    r_last = pose_format(fa.get_pose())
    t_start = time.time()
    t_last = t_start
    t_ready = t_start
    i = 0
    print("time begins at: ",t_start)
    # ================================================while begins
    while not rospy.is_shutdown():
        # rospy.loginfo("start while-----")

        t = time.time()
        q_now = fa.get_joints()
        qdot = fa.get_joint_velocities()
        
        if len(q_now)==0:
            print("can not get q !!!!!")
            continue
        # x_now = vision_reader.pos
        # x = x_now
        # k_now = vision_reader.k_pos
        r_now = pose_format(fa.get_pose())
        r = r_now


        # ????????????????????? J
        J = fa.get_jacobian(q_now)  # (6, 7)
        J_inv = np.linalg.pinv(J)  # (7, 6)


        pose = fa.get_pose().matrix
        J_ori_inv = np.linalg.pinv(J[3:6, :]) # only compute orientation!!
        J_pos_inv = np.linalg.pinv(J[0:3, :])
        N = np.eye(7) - np.dot(J_inv, J)
        N_ori = np.eye(7) - np.dot(J_ori_inv, J[3:6, :])
        N_pos = np.eye(7) - np.dot(J_pos_inv, J[0:3, :])


        # ?????????
        if flag_initialized == 0:# ?????????????????????
            fa.reset_joints(block=True)
            time.sleep(1)
            flag_initialized = 1

        elif flag_initialized == 1:
            home_joints = fa.get_joints()

            max_execution_time = 30

            fa.dynamic_joint_velocity(joints=home_joints,
                                    joints_vel=np.zeros((7,)),
                                    duration=max_execution_time,
                                    buffer_time=10,
                                    block=False)
            
            flag_initialized = 2
            print("initialization is done! the time is:", t - t_start)
            t_ready = t
            # fa.run_guide_mode(duration=20,block=False)

            # RR = np.array([[-7.34639719e-01, -6.27919077e-04,  6.78457138e-01], \
            #                [-6.78432812e-01,  9.19848654e-03, -7.34604865e-01], \
            #                [-5.77950645e-03, -9.99957496e-01, -7.18357448e-03]]) # ???????????????base???????????????,????????????base?????????????????????
            # RR = RR.T
            # f = 2340
            # u00 = 753
            # v00 = 551
            # x = np.reshape(x, [2, 1])
            # theta_k0 = np.array([[f * RR[0, 0]], \
            #                      [f * RR[0, 1]], \
            #                      [f * RR[0, 2]], \
            #                      [f * RR[1, 0]], \
            #                      [f * RR[1, 1]], \
            #                      [f * RR[1, 2]], \
            #                      [RR[2, 0]],     \
            #                      [RR[2, 1]],     \
            #                      [RR[2, 2]],     \
            #                      [u00 * RR[2, 0]], \
            #                      [u00 * RR[2, 1]], \
            #                      [u00 * RR[2, 2]], \
            #                      [v00 * RR[2, 0]], \
            #                      [v00 * RR[2, 1]], \
            #                      [v00 * RR[2, 2]]])
            # theta_k = theta_k0

        elif flag_initialized == 2 and t - t_ready < 20: 
            # # ??????????????????Js
            # u = x[0] - u0
            # v = x[1] - v0
            # z = 1 # =========================
            # Js = np.array([[fx/z, 0, -u/z], \
            #                [0, fy/z, -v/z]])
            # Js = np.dot(Js, RR)
            # Js_inv = np.linalg.pinv(Js)

            # Js_hat = np.array([[theta_k[0,0]-theta_k[6,0]*x[0,0]+theta_k[9,0], theta_k[1,0]-theta_k[7,0]*x[0,0]+theta_k[10,0], theta_k[2,0]-theta_k[8,0]*x[0,0]+theta_k[11,0]],\
            #                                                 [theta_k[3,0]-theta_k[6,0]*x[1,0]+theta_k[12,0], theta_k[4,0]-theta_k[7,0]*x[1,0]+theta_k[13,0], theta_k[5,0]-theta_k[8,0]*x[1,0]+theta_k[14,0]]])

            # ???????????? = r_now
            

            # ??????????????????
            # rdot = np.dot(J, np.reshape(qdot, [7, 1]))
            rdot = np.dot(J,np.reshape(qdot, (7,1)))

            # ??????ut
            ut = - np.dot(J_inv, np.dot(Kp, error_format(r,r_d)))
            print("error:",np.reshape(error_format(r,r_d),(6,)))

            # # ??????un
            # if t - t_ready > 15 and t - t_ready < 16:
            #     d = np.reshape(np.array([-0.2, -0.2, 0.2, 0.2, 0.2, 0.1, 0.1], float), [7, 1])
            #     un = -np.dot(N_pos, np.dot(np.linalg.inv(Cd), d)) 
            # else:
            #     un = np.zeros([7,1])
            

            
            v = ut 
            v = np.reshape(np.array(v), [-1,])
            
            v[v>0.5] = 0.5
            v[v<-0.5] = -0.5
            
            log_r.append(r.tolist())
            log_q = np.concatenate((log_q,np.reshape(q_now, [7, 1])), axis=1)
            log_qdot = np.concatenate((log_qdot,np.reshape(qdot, [7, 1])), axis=1)
            log_dqdot = np.concatenate((log_dqdot,np.reshape(v, [7, 1])), axis=1)
            log_rdot = np.concatenate((log_rdot,np.reshape(rdot, [6, 1])), axis=1)

            traj_gen_proto_msg = JointPositionVelocitySensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - t_ready, 
                seg_run_time=30.0,
                joints=home_joints,
                joint_vels=v.tolist()
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION_VELOCITY)
            )
            
            i += 1
            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
            pub.publish(ros_msg)


        else:
            break

        q_last = q_now
        # k_last = k_now
        r_last = r_now
        t_last = t

        rate.sleep()# Sleeps for any leftover time in a cycle. Calculated from the last time sleep, reset, or the constructor was called. ????????????ros_freq Hz
    # ===============================================while ends
    # print(time.time()-t_start)

    # print(np.shape(log_qdot))
    # log_x_array = np.array(log_x)

    # np.save(current_path+ dir+'log_x.npy',log_x_array)
    # np.save(current_path+ dir+'log_q.npy',log_q)
    # np.save(current_path+dir+'log_qdot.npy',log_qdot)
    # np.save(current_path+dir+'log_rdot.npy',log_rdot)
    # np.save(current_path+dir+'log_dqdot.npy',log_dqdot)

    # # task space velocity==============================================
    # plt.figure(figsize=(30,20))
    # for j in range(6):
    #     ax = plt.subplot(3, 2, j+1)
    #     ax.set_title('task space velocity %d' % (j+1),fontsize=20)
    #     plt.xlabel('time (s)')
    #     if j<3:
    #         plt.ylabel('velocity (m/s)')
    #     else:
    #         plt.ylabel('angular velocity (rad/s)')

    #     plt.plot(np.linspace(0,np.shape(log_rdot)[1]/ros_freq,np.shape(log_rdot)[1]),np.reshape(np.array(log_rdot[j,:]),[-1,]) ,label = 'actual veloc')
    #     plt.legend()
    # plt.savefig(current_path+ dir+'log_r.jpg')

    # # vision space position===============================================
    # plt.figure()
    # plt.plot(log_x_array[:,0], log_x_array[:,1],label = 'actual')
    # plt.scatter(dx[0],dx[1],label = 'target', c='r')
    # plt.legend()
    # plt.title('vision space trajectory')
    # plt.xlabel('x (pixel)')
    # plt.ylabel('y (pixel)')
    # plt.savefig(current_path+ dir+'log_x.jpg')

    # # vision space position verse time======================================
    # fig = plt.figure(figsize=(20,8))
    # plt.plot(np.linspace(0,np.shape(log_rdot)[1]/ros_freq,np.shape(log_rdot)[1]), log_x_array[:,0]-dx[0],label = 'x')
    # plt.plot(np.linspace(0,np.shape(log_rdot)[1]/ros_freq,np.shape(log_rdot)[1]), log_x_array[:,1]-dx[1],label = 'y')
    # plt.legend()
    # # plt.title('vision space error')
    # plt.xlabel('time (s)')
    # plt.ylabel('error (pixel)')
    # plt.savefig('log_x_t.jpg',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)


    # joint space velocity=================================================
    plt.figure(figsize=(30,20))
    for j in range(7):
        ax = plt.subplot(4, 2, j+1)
        ax.set_title('joint space velocity %d' % (j+1),fontsize=20)
        plt.xlabel('time (s)')
        plt.ylabel('velocity (rad/s)')

        plt.plot(np.linspace(0,np.shape(log_qdot)[1]/ros_freq,np.shape(log_qdot)[1]),np.reshape(np.array(log_qdot[j,:]),[-1,]) ,label='actual joint velocity')
        plt.plot(np.linspace(0,np.shape(log_dqdot)[1]/ros_freq,np.shape(log_dqdot)[1]),log_dqdot[j,:].reshape(-1,), label = 'command joint velocity')
        plt.legend()
    plt.show()
    # plt.savefig(current_path+ dir+'log_qdot.jpg')

if __name__ == '__main__':
    main()
