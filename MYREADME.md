# IRM Lab Version
*modified by jyp, yxj*


### Log
* **0415**
    * finished omega 3 trajectory recording. `examples/my_haptic_subscriber.py`
    * finished trajectory recording and dmp reproducing. `examples/my_run_joint_dmp.py`
    * ***record trajectory***:
        1. Modify necessary parameters in `my_haptic_subscriber.py`
            | parameter | location | definition |
            | :----: | :----: | :----: |
            | FILE_NAME | line 24 | trajectory save path |  
           Modify other parameters in `my_haptic_subscriber.py`, including `width_y`, `angle_y`, `haptic_scale`, `record_time` and `record_rate`
        2. Start Franka, Omega3 and controll PC(run the bash scripts in frankapy)
        3. run
            ```python
                python ./examples/my_haptic_subscriber.py
            ```
        4. Wait until Franka gripper opens, and you have `3 second` to let it grab something
        5. Make sure Omega3 is at its home position and DO NOT MOVE Omega3 until it tells you to do so in the terminal.
        6. Slowly moves Omega3 and guide the robot to goal

* **0416**
    * tested dmp goal position change
    * finished reset arm after Franka released the object at goal position
    * ***record trajectory***:
        1. Modify necessary parameters in `my_run_joint_dmp.py`
            Modify file path in line 20 and other parameters at the beginning of main().
            Carefully check if `record_frequency`, `execute_frequency` and `tau` are correct!
        2. Start Franka and controll PC(run the bash scripts in frankapy)
        3. run
            ```python
                python ./examples/my_run_joint_dmp.py
            ```
        4. Wait until Franka gripper opens, and you have `3 second` to let it grab something
        5. The robot will reproduce recorded trajectory with joint dmp, released the grasped object, and go back to home position.

* **0417**
    * To avoid virtual wall collision warning, comment line 38~45 in franka-interface: termination_handler.cpp
    * added Cartesian translation and rotation velocity limit in my_haptic_subscriber.py

* **0423**
    * 在Frankapy中添加新的Skill，利用此Skill创建新的任务函数  
        * 在`franka_interface_common_definitions.py`中选择`SkillType`,`TrajectoryGeneratorType`,`FeedbackControllerType`,`TerminationHandlerType`类型。允许自定义新的类型，命名与franka-interface的`definition.h`保持一致，参考后面部分的介绍  
        * 在`franka_arm.py`中创建新的任务函数`def execute_XXX()`，定义包含所选择`SkillType`,`TrajectoryGeneratorType`,`FeedbackControllerType`,`TerminationHandlerType`，例如  
        ```
            skill = Skill(SkillType.MyJointVelocitySkill, 
                      TrajectoryGeneratorType.MyJointVelocityGenerator,
                      feedback_controller_type=FeedbackControllerType.NoopFeedbackController,
                      termination_handler_type=TerminationHandlerType.TimeTerminationHandler, 
                      skill_desc=skill_desc)
        ```
        * 参考`franka_arm.py`中其他任务函数调用skill的相关方法，如`add_initial_sensor_values()`,`set_joint_impedances()`,`add_goal_joints()`,最关键的是`skill.create_goal()`和`self._send_goal()`，之后frankapy将与franka-interface通信，执行任务
    * 在Frankapy中添加新的dynamic control任务  
        * 在dynamic control任务中可以向Franka实时发送joint space或cartesian space的position,velocity实现控制  
        * 参考`./examples/run_dynamic_joints.py`,`./run_dynamic_pose.py`,之后定义新的dynamic control任务  
            1. dynamic control任务需要一个发布position,velocity的publisher  
            2. 执行franka_arm中的某个任务函数，如`fa.goto_joints`，设置参数`dynamic=True`及`block=False`   
            3. 创建新的dynamic control消息，如`JointPositionSensorMessage`，其他消息类型见`./proto/sensor_msg_pb2.py`，消息类型包含若干字段，详见此文件前半部分的描述。填入各字段，不能有空字段  
            4. `make_sensor_group_msg`将dynamic control消息封装为rosmsg，1个参数指明此rosmsg属于Skill中3类任务的哪一个，其中`SensorDataMessageType`需要与定义的dynamic control消息类型保持一致  
            5. 将rosmsg发布到`FC.DEFAULT_SENSOR_PUBLISHER_TOPIC`话题上，注意设置发布的rate  
            6. 任务结束后，发布termination_message，参考`./examples/run_dynamic_joints.py`,`./run_dynamic_pose.py`
    * 在franka-interface中添加新的trajectory generator
        * 我们添加了`my_joint_velocity_generator`，可以通过Frankapy中的dynamic control任务`dynamic_joint_velocity`动态控制joint velocity
        * `my_joint_velocity_generator`参考了`joint_trajectory_generator`，添加的文件包括`my_joint_velocity_generator.h/.cpp`，修改的文件包括`trajectory_generator_factory.cpp`,`/franka-interface-common/include/franka-interface-common/definitions.h`.后者文件要和`frankpy/franka_interface_common_definitions.py`在内容、顺序上都一致。
        * 在写`my_joint_velocity_generator.cpp`时，注意
            * switch(skill_type)要添加自己写的skilltype，例如`SkillType::MyJointVelocitySkill`.
            * `parse_parameters`,`get_next_step`在基类`TrajectoryGenerator`中被定义为纯虚函数,一定要在派生类中定义
        * 改完franka_interface记得编译。先`make_franka_interface.sh`再`make_catkin.sh`

* **0429**
    * 代码文件更名：  
        * `my_Cartesian_vision_based_control.py`：给定Franka EE在Cartesian space的位姿(6DOF), 通过关节速度控制(P控制)到达目标  
        * `my_vision_based_control.py`：(目前)给定图像平面目标点$x_d$, 通过关节速度控制(P控制)到达目标  
        * `my_vision_based_control_with_dmp.py`：是`my_vision_based_control.py`加上dmp的版本。根据图像空间的目标位置抓取物体后，reset到home position，执行dmp（其中goal position改过了，和专家轨迹的goal不同）放置物品
    * 更新了`my_vision_based_control.py`的控制律中需要用到的Jacobi矩阵$J$：  
        * 在Franka夹爪侧面粘贴ArUco码(Id=99, Size=0.039*0.039m, 注意实验室HP打印机打出来的码比标称值要小), 用于Franka在相机中定位  
        * 利用ROS的tf机制标定ArUco码相对Franka EE的位姿  
            1. 在`start_vision_based_control.launch`中添加了`panda_moveit_config panda_control_moveit_rviz.launch`的启动代码。于是, tf中会发布Franka各link以及EE的tf树  
            2. 在`start_vision_based_control.launch`中添加了发布静态坐标变化的代码，利用`static_transform_publisher`实现。于是, tf中会发布panda_link0 $\rightarrow$ camera_link的坐标变换（即$_{camera\_link}^{panda\_link0}T$,也就是我们平时说的camera_link到panda_link0，注意tf的顺序是反的），即camera的外参  
            3. 在`./examples/marker_tf_listener.py`中，订阅`/aruco_single/transform`，将其中ArUco码在camera frame中的位姿发布到tf上。并通过`TransformListener`的`lookupTransform`查看ArUco marker到Franka EE的tf变换  

            ***NOTES***  
            * 在tf中，利用`TransformBroadcaster`和`TransformListener`来发布和订阅tf变换。tf中参数顺序在前的为`parent frame`，在后的为`child frame`。在tf树中，箭头从`parent frame`指向`child frame`，对应的齐次变换矩阵为$_{child\_frame}^{parent\_frame}T$，**容易颠倒，务必注意!!!**  
            * `start_vision_based_control.launch`中启动panda_moveit_config，不能与frankapy或Franka Interface同时运行，否则Franka Interface会挂起，容易导致死机。猜测Franka Interface底层采用了mutex内存保护机制(*reference:*[Mutex类文档]https://docs.microsoft.com/zh-cn/dotnet/api/system.threading.mutex?view=net-6.0)  

            4. 以上只是得到ArUco marker到Franka EE的tf变换，进一步还需得到joint velocity $\rightarrow$ ArUco marker在Cartesian space速度的Jacobi矩阵


### Warning
1. The quaternion representation is different in scipy and RigidTransform, convertion is needed!
    * In scipy: `[x, y, z, w]`
    * In RigidTransform: `[w, x, y, z]`
2. RigidTransform is implemented with shallow copy in low-level!
    * Use `copy.deepcopy(...)` when copying RigidTransform objects, or even its attributes.
3. ROS topics use queue to store coming messages!
    * To acquire real-time data in callback function, modify the queue size of both subscriber and publisher to 1.
    * And modify the buffer length of subscriber to a large number, or leave it as default.
4. Pay ttention to numpy matrix multiply!
    * `*` means element wise multiplication
    * `np.matmul(a, b)` means matrix multiplication
5. When assert is triggered in callback function of ROS subscriber, the message processing slows down, and then some messages might be discarded when the subscriber queue is full
