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
        2. Start Franka, Omega 3 and controll PC(run the bash scripts in frankapy)
            * starting Omega 3: first `cd /dev/bus/usb/001`; second `lsusb` to find the file corresponding to Omega 3, let's say `005`; third `sudo chmod 777 005`; last `rosrun haptic_ros_driver haptic_ros_driver`.
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
* **0503**
    * 完成了`my_haptic_subscriber_hololens.py`. 
        * 基本思路是从`my_haptic_subscriber.py`开始改，把里面的笛卡尔空间阻抗控制器换成关节空间速度控制器。原先需要的proto_msg有`PosePositionSensorMessage`和`CartesianImpedanceSensorMessage`，相当于笛卡尔空间阻抗控制器的参数通过`CartesianImpedanceSensorMessage`传到底层franka_interface去计算。现在控制器全部在上层写了，需要的proto_msg只有`JointPositionVelocitySensorMessage`一个.
        * 在`FrankaArm`中添加了`get_jacobian_joint4()`用于将hololens的输入转化成`d`
        * 参数有`Kp`和`Cd`，现在都是单位阵，速度限制设为0.3。这些参数都是调过的，基本够用。
        * 使用前，先更改`record_time`
        * 注意：如果发现Nullspace导致实际末端位姿有变化，很有可能是速度限幅将某几个关节限制了。
    * 完成了和hololens的联调
        ```
        bash bash_scripts/start_control_pc.sh -i localhost
        roslaunch ros_tcp_endpoint endpoint.launch
        rosrun haptic_ros_driver haptic_ros_driver
        # activate virtual environment and source
        python ./examples/my_haptic_subscriber_hololens.py
        ```

* **0517**
    * 完成了region control部分关于region的定义，见`my_adaptive_control.py`中`ImageSpaceRegion`、`CartesianSpaceRegion`、`CartesianQuatSpaceRegion`、`JointSpaceRegion`，和论文中的large scale控制律`AdaptiveImageJacobian`
    * 在gazebo中搭建仿真环境，调试`JointSpaceRegion`的控制效果。设置一个目标region控制Franka在关节空间接近目标；同时设置一个奇异点region控制Franka远离奇异位置, gazebo中的测试代码mian()为test_joint_space_region_control, 只考虑joint space的控制律为`JointOutputRegionControl`  
      * 设Franka的Body Jacobian为$J_b$, 则用manipubility描述机械臂运动学奇异程度$\mu(J_b)=\sqrt{det(J_b{{J_b}^T})}$  
      * 测试结果见`/data/0517/test_joint_space_region.png`, 左图为机械臂运动过程中, 在关节空间距离奇异点的2-范数距离, 右图为机械臂运动过程中的manipubility变化  
      * gazebo仿真环境搭建, 主要利用`franka_ros`包里的`franka_gazebo`和`franka_example_controllers`两个package实现, 目前修改了`franka_example_controllers::joint_velocity_example_controller`。具体修改相当于添加了若干subscriber和publisher，通过subscriber接收topics上的关节速度控制指令，通过publisher发布Franka的body jacobian、joint angle、joint velocity、end effector pose等一系列指令。请注意，各franka_example_controller继承自controller_interface::MultiInterfaceController，使用相关hardware interface，请修改controller_interface::MultiInterfaceController的模板参数，否则通过`robot_hw->get`索取hardware interface可能会返回nullptr
    * 经过测试发现，假设body jacobian为$J_b$，其伪逆${J_b}^+$，则控制律中与joint space region相关的分量$u=-{J_b}^+{J}\xi_{q}$
    * ***Tutorial for Gazebo Simulation***
      1. Clone the `franka_ros` package from github into your workspace, like `franka_ws/src`.
        ```
            git clone https://github.com/Director-of-G/franka_ros.git
        ``` 
      2. When you modify anything in `franka_ros`, remember to `catkin_make` the workspace.
      3. Clone the `frankapy` package from github.
        ```
            git clone https://github.com/Director-of-G/frankapy.git
        ```
      4. Add the following line into `~/.bashrc`.
        ```
            source ~/franka_ws/devel/setup.bash
        ```
      5. Launch the Franka Gazebo environment.
        ```
        roslaunch franka_gazebo panda.launch x:=-0.5 world:=$(rospack find franka_gazebo)/world/stone.sdf controller:=joint_velocity_example_controller rviz:=true
        ```
      6. Launch the frankapy controller in `./examples/gazebo/my_gzb_adaptive_control.py`. Make sure the test code `test_joint_space_region_control()` is not commented in main().
        ```
            cd /path/to/frankapy
            python ./examples/gazebo/my_gzb_adaptive_control.py
        ```
      7. To add joint regions, it is recommended to use `JointSpaceRegion.add_region_multi()` in `./examples/gazebo/my_gzb_adaptive_control.py`, the following parameters should be included.
           * **qc**: np.ndarray -- region center in joint space (radians, shape=(7,))
           * **qbound**: float -- radius of the joint space region(hyper-sphere of hyper-ellipsoid)
           * **qrbound**: float -- radius of the joint space reference region(hyper-sphere of hyper-ellipsoid)
           * **mask**: np.ndarray -- which joint are considered in the joint space region (eg. np.array([1, 1, 1, 1, 1, 1, 1]) means all joints are considered, while np.array([1, 0, 1, 1, 0, 1, 1]) means $joint1$ and $joint4$ are excluded)
           * **kq**: float -- parameter $k_{qi}$
           * **kr**: float -- parameter $k_{ri}$
           * **inner**: bool -- whether the region should be reached (`inner=False`) or kept away from (`inner=True`)
           * **scale**: np.ndarray -- scale factor of different joints (eg. np.array([1, 1, 1, 1, 1, 1, 1]) means the hyper-sphere joint region, while np.array([1, 2, 1, 0.5, 1, 0.25, 4]) means the hyper-ellisoid joint region)

* **0526**
    * 在Gazebo中搭建好Journal实验所需的仿真环境，包括Franka Panda Arm、相机、Franka Gripper和抓取物体(带有ArUco码)
    * ***Tutorial for Gazebo Simulation***
      1. `git clone` or `git pull` the `franka_ros` package from github into your workspace, like `franka_ws/src`.
        ```
            git clone https://github.com/Director-of-G/franka_ros.git
        ``` 
      2. `git clone` or `git pull`  the `frankapy` package from github.
        ```
            git clone https://github.com/Director-of-G/frankapy.git
        ```
      3. ArUco markers are rendered in Gazebo as types of material, declared through the \<material\> label. Thus a material description script and pictures of the material are needed, which are provided in `/path/to/franka_ros/utils`. Note the path `/usr/share/gazebo-9/media/materials` as `<gazebo_prefix>`. We should copy the files as follows
        ```
            sudo cp /path/to/franka_ros/utils/aruco_98.png <gazebo_prefix>/textures/aruco_98.png \
            sudo cp /path/to/franka_ros/utils/aruco_123.png <gazebo_prefix>/textures/aruco_123.png \
            sudo cp /path/to/franka_ros/utils/aruco.material <gazebo_prefix>/scripts/aruco.material
        ```
      4. Modify file `double.launch` in the `aruco_ros` package. Six lines need to be changed.
        ```
            <arg name="marker1Id"         default="98"/>
            <arg name="marker2Id"         default="123"/>
            <arg name="markerSize"        default="0.04"/>    <!-- in m -->
            <arg name="dct_normalization" default="False" />
            <remap from="/camera_info" to="/franka/camera1/camera_info" />
            <remap from="/image" to="/franka/camera1/image_raw" />
        ```
      5. Launch the Gazebo simulation environment.
        ```
            roslaunch franka_gazebo panda.launch x:=-0.5 \
              world:=$(rospack find franka_gazebo)/world/stone.sdf \
              controller:=cartesian_impedance_example_controller \
              rviz:=true
        ```
      6. Launch the aruco_ros node to detect two markers simultaneously.
        ```
            roslaunch aruco_ros double.launch
        ```
      7. *TODO*  
        * Add publishers `/aruco_simple/pixel1` and `/aruco_simple/pixel2` in `simple_double.cpp` for two markers centers in the image plane.
        * Add code to test region control in robot grasping scene.
        * Add code to test adaptive NN in robot grasping scene.

* **0607**
    * 完成了`my_joint_region.py`.是从`my_gzb_adaptive_control.py`复制过去改的，把原本`my_gzb_adaptive_control.py`中的data_collection删掉了，换上了实物的fa来替代其功能

* **0611**
    * 完善了昨天的`my_cartesian_region.py`.但是发现效果一直不明显，要么就震荡。一般应该qrbound比qbound更大，之前yp设的参数会导致明显震荡。参数和效果记录在`log_parameters.odt`中

* **0617**
    * 对于**0526**的*TODO*，1已经完成了，经过验证，图片显示的立方体框会往右偏大约200个像素，但是`/aruco_simple/pixel1`等topic中的数值是正确的数值。
    * 需要写一个控制器，让它一开始就移到视野范围内.(finished)
    * 相机内参的设定都写在`franka_description`下的`panda_gazebo.xacro`,但不是直接写内参矩阵，内参矩阵要通过`rostopic echo camera_info`来得到.
    注意目前启动gazebo环境的指令是```roslaunch franka_gazebo panda.launch x:=-0.5 world:=$(rospack find franka_gazebo)/world/stone.sdf controller:=joint_velocity_example_controller rviz:=true```
* **0618**
    * 完成了**0526**的*TODO*中的2的vision region验证部分。代码是`my_gzb_adaptive_control.py`中的`test_vision_joint_space_region_control`,数据和画图程序都放在`data/0618`
    * `examples`文件夹下的带有`my_yxj_test`的都是我的随意测试的代码，没什么用，不用管它们
* **0623**
    * 在`my_gzb_adaptive_control.py`中新增了两个类：`MyConstantsSim`和`AdaptiveImageJacobianSim`。对于原来的`AdaptiveImageJacobian`，个人感觉里面不应该放各种`region control`控制器，于是新建了针对仿真中`Js`更新的`AdaptiveImageJacobianSim`.测试代码是函数`test_adaptive_region_control()`
* **0628**
    * 出校着手实物实验。
    * 几乎完成不使用ANN时的`cartesian, vision and joint region`的实物实验。能跑但是有几个问题：
        * 没法抓住。视觉空间确实能到达目标位置，但是深度全过程中几乎不变，要么碰到要么太高。
        * Franka在路径附近本质上是没有`singularity`的。
            * 现在想到的办法是让第一二个link尽量面向正前方？
        * 相机识别有几个瞬间会不准
            * 需要换大一点的`aruco`码

* **0630**
    * 改了`ws_franka`里面的`double.launch`及其相关源文件。现在机械臂上的码是99，目标码是485.
    * 完成了`my_precise_control.py`.数据存在文件夹`0630`中.跑之前记得打开相机：
        ```
        (in workspace ws_franka)
        roslaunch pylon_camera pylon_camera_node.launch 
        roslaunch aruco_ros double.launch
        rosrun rqt_image_view rqt_image_view 

        (in frankapy)
        bash bash_scripts/start_control_pc.sh -i localhost
        (in environment py36)
        python examples/my_precise_control.py
        ```

* **0701**
    * 完成ANN+vision+Cartesian region在Gazebo中的验证
    * 跑出一组参数，在有ANN和无ANN生效的情况下，结果有明显对比
    * 参数如下：
      * 3个维度上RBF中心的取值范围：[[-0.3, 0.7], [-0.3, 0.7], [0, 1]] (每维10个；现在的取值范围偏大，尝试让取值范围缩小到FOV中，效果不好，收敛但有静差)
      * 3个维度上RBF中心$\sigma$的值：1(若取$\sigma=\frac{D}{\sqrt{K}}$，让相邻RBF的overlap不超过1倍$\sigma$，效果不好，收敛速度略慢于无RNN情形，但也有超调)
      * 目前初始的$J_s$给的是有偏差的，其中参数$f_x=3759.66467+200,f_y=3759.66467-200,u_0=960.5-50,v_0=540.5+50$，但原始的$J_s$本就不精确(它的假设是marker中心位于夹爪中心，但仿真和实物都有offset，不过利用目前不精确的$J_s$，实验效果还可以)
    * 代码在./gazebo/my_gzb_adaptive_control.py中的test_adaptive_region_control函数
      * 唯一参数allow_update=True/False，表示是否运用ANN更新参数
      * 在main中运行test_adaptive_region_control即可(默认不使用ANN)，需要开gazebo环境和aruco_ros
    * 目前的对比效果是
      * 使用ANN，收敛较慢，但几乎无超调
      * 不使用ANN，收敛较快，超调明显
    * 数据在`./data/0701`下，`./with adaptive`和`./with no adaptive`两个文件夹分别包含使用ANN和不使用ANN下的实验结果，能够看到上述对比效果。运行test_adaptive_region_control，会绘图显示更新后的权重矩阵$W$，并在`./data/0701`中给出`data_with_adaptive.pkl`和`data_with_no_adaptive.pkl`文件记录数据，注意修改存放记录的文件夹名称。

* **0702**
    * 修改了AdaptiveRegionControllerSim中的Image Jacobian，即图像平面和Marker的Body Twist空间的投影矩阵
    * 目前还没有找到一组合适的参数，让有adaptive和无adaptive时有明显差异
    * 图像平面marker到目标点在x和y两个维度上收敛的速度可能有差异，导致x或y坐标之一先收敛，另一者随后收敛，marker中心在图像平面的轨迹未必是直线，可以在`set_Kv(np.array([[0.2, 0.1]]))`中调节不同维的增益参数
    * 使用修改后的$Js$，需要额外做几件事情：
      * `AdaptiveRegionControllerSim`的`get_u`和`update`需要传入参数`p_s`，他是夹爪中心到marker中心的translation在world坐标系中的值，只要测量该translation在`panda_EE`中的值，参考`test_adaptive_region_control`中`get_u`和`update`两个函数前的`# 示例：计算p_s`的操作即可。注意`AdaptiveRegionControllerSim`的`__init__`中也需根据实机修改`p_s`
    * 完成`my_adaptive_control.py`,但是哪怕是准确的Js不更新也会分段到达图像空间中的目标点，这是不应该的，如果Js准确应该笔直到达目标点。应该是marker的位置不是夹爪的问题。

* **0704**
    * 发现0702的问题是和orientation region有冲突了，换了一个目标角度就没问题了
    * 问题1： 不同region的项抵消的问题
    * 问题2： ann效果不明显的问题

* **0709**
    * `my_test_image_jacobian.py`用来验证Js的正确性。现在看来是有点问题的。
    * 为啥yp会说`Js@Js.T`这个2x2矩阵是单位阵？他明明是`[[3544042.44830763   28420.30210352]
 [  28420.30210352 3600635.57085787]]`这样的。难道说经过单位的转化之后是单位阵？那应该怎么转换呢？
    * `my_gzb_adaptive_control.py`中`dq_d = J_pinv @ (Js_hat.T @ x_dot)`和kesi_x无关,yp为什么说`速度和kesi_x共线`
    * 明天1.我再解决一下这个Js为什么不对，为什么仿真中是ok的？2.把矢量图画好。

* **0710**
    * 编写了`image Jacobian`的调试节点，源代码位于`./examples/gazebo/image_debug.py`
    * 该文件运行名为`image_debugging`的ROS节点，该节点将当前ArUco Marker中心处计算出的`-kesi_x`方向(绿箭头)，marker在像素平面期望速度方向(红箭头)，和marker的实际速度方向(蓝箭头)绘制在图像中。图像左上角显示的是kesi_x与期望速度比值，用于验证二者间的变换矩阵是否接近单位阵
    * 使用时需要将`ImageDebug`类的`pose_sub`对象用`frankapy`的接口代替
    * 运行方式
      * `python ./examples/gazebo/image_debug.py`
      * `rosrun rqt_image_view rqt_image_view`查看发布在`/image_debug/result`的图像消息即可
    * 报错
      * Error1
        ```
        ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)
        ```
        ROS自带的cv_bridge只能用python2环境运行，用python3执行需要自行编译cv_bridge包，考虑到项目文件包含较多python3特定语法，一种快速解决方式参考[CSDN](https://blog.csdn.net/qq_43124746/article/details/124347630)

* **0712**
    * 完成`my_adaptive_control_0712.py`。加入了Js的debug功能(也就是加入了`my_image_debug_real.py`的内容,但是`my_image_debug_real.py`不能单独跑,因为fa只能传入一个程序,无法将`my_image_debug_real.py`和`my_adaptive_control.py`同时跑起来,遂写了`my_adaptive_control_0712.py`)。发现效果还好，只是蓝和红有些不共线。
        跑之前记得
        ```
        source ~/python3_ws/devel/setup.bash --extend
        ```
    * 限制某些关节进入一些joint region还有问题。

* **0714**
    * 加上joint region。注意`add_region_single`是为了1自由度机械臂写的，目前已经没用了。对某一个关节加region还是要用`add_region_multi`,只不过其中的mask设置一下就可以了。

* **0716**
    * 完成了加入单个joint排斥region的实验,代码是`my_joint_region.py`,数据在`0716/my_joint_region/`中.虽然从关节角度上joint 4的效果不明显,但是`kesi_q`和`dq_d`的差别还是很大的.(原先写的`kesi_q`函数有点问题,多个region的时候计算会出错,问题出在sum的位置不对.现在已经debug好了.)
    * 修改`my_adaptive_control_0712.py`代码,
        * 将其中joint region中的`add_region_single`全部去掉了(代码`my_joint_region.py`和`my_adaptive_control.py`中也去掉了),
        * 'kesi_q'已经改成正确的了.(虽然本实验中只有一个region不会出问题)

* **0717**
    * 完成了相机视角视频记录的代码`save_video.py`，放在包`pylon_camera`中.命令：
        ```
        roslaunch pylon_camera camera_aruco_view.launch
        roslaunch pylon_camera camera_aruco_recording.launch
        ```

* **0718**
    * 完成了'my_whole_task.py'
    * 对`my_adaptive_control_0712.py`进行debug，发现之前蓝红不共线的原因是`image_debug`类中的`Kv`和实际的`Kv`方向不同导致的。改了之后就共线了。
    * 用`my_adaptive_control_0712.py`进行了3组实验，数据在`data\0718\`中。分别是`Js_wrong`,`Js_real`,`Js_update`.

* **0719**
    * dirty method for `my_adaptive_control_0712.py`: 笛卡尔空间y方向会绕一个弯，解决办法：将Ko调小，即让方向收敛比位置收敛慢。（新的组合方法之后再思考）

* **0720**
    * 进一步修改了`my_haptic_subscriber_hololens.py`。目前nullspace只针对末端位置，这样就能用hololens微调末端的角度，不影响omega决定的末端位置。

* **0721**
    * 完成了exp1。demonstration的代码是`my_haptic_subscriber_hololens.py`,数据存在`0721`文件夹中，画图文件是`0720/draw0721.py`。execution的代码是`my_run_joint_dmp.py`

* **0723**
    * `adaptive control`中,更新Js的代码实测执行较慢,瓶颈在于计算dW_hat的矩阵运算过程,每次大约需要0.2~0.3s. 这部分代码在单独的python文件运行,每次计算dW_hat也只需要约0.002s,即使`adaptive control`在同时运行.推测是操作系统对单个进程内存分配(速度)限制的问题?
    * 尝试使用`Multiprocessing.Pool`为计算矩阵运算的代码开启单独的进程,效果不明显,矩阵运算耗时约0.15s?
    * 下午通过某种代码的奇妙组合,矩阵运算的速度突然加快了,目前0.03~0.05s左右,17:00左右push到git仓库,并跑出一组理想的实验数据`/data/0723/my_adaptive_control_20220723_163653_with_Js_no_update`

    * 晚上调出了有自适应收敛，无自适应不收敛的实验效果。有自适应收敛的效果是`my_adaptive_control_20220723_221648_1`，无自适应的效果是`my_adaptive_control_20220723_221313_-1`
    去掉joint region的实验效果是`my_adaptive_control_20220723_223953_1`,效果并不好，不收敛（原因是没有joint region之后导致进入FOV姿势很正，最终导致kesi_x和orientation region抵消）
    
    
### 常用命令
1. 启动仿真环境
   * 启动Gazebo
   ```
   roslaunch franka_gazebo panda.launch x:=-0.5 world:=$(rospack find franka_gazebo)/world/stone.sdf controller:=joint_velocity_example_controller rviz:=true
   ```
   * 启动aruco_ros节点
   ```
   roslaunch aruco_ros double.launch
   ```
   * 启动rqt_image_view
   ```
   rosrun rqt_image_view rqt_image_view
   ```
2. 跑仿真环境的控制器
   ```
   cd ~/franka_ws/src/frankapy
   python ./examples/gazebo/my_gzb_adaptive_control.py
   ```
3. 启动Tensorboard
   ```
   cd ~/franka_ws/src/frankapy
   tensorboard --logdir=./data/<日期>/<文件名> --host=127.0.0.1 --port=8008
   ```

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
