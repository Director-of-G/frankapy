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
