#!/usr/bin/env python3
"""ROS node for publishing predicted trajectories based on camera and joint state inputs."""

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
import time
from collections import deque      # 线程安全、无锁的双端队列
from threading import Lock
import math

# Constants for configuration
JOINT_COUNT = 7
RESET_POSITION = [0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.08]  # Resetting to 0
RESET_VELOCITY = [10.0] * JOINT_COUNT  # Velocity to zero after stopping
RESET_EFFORT = [10.0] * JOINT_COUNT  # Effort to zero after stopping
RATE_HZ = 20  # Set the desired frequency for the main loop
MAX_TRAJECTORY_INDEX = 1000  # Limit for the trajectory updates
lock = Lock()


trajectory={}; 
cnt=0;

def callback(msg):
    global cnt;
    cnt+=1;
    global trajectory, write_idx
    stateTimestamp=msg.data[-1]; #  从msg中获取stateTimestamp
    arr = np.array(msg.data[:-1]).reshape(-1, 14)

    import math

    L = arr.shape[0]            # 一般是 50
    sigma = L / 4.0             # 调得更大/更小可改变平滑度
    new_traj = 0
    for i in range(L):
        key = int(stateTimestamp * RATE_HZ + i) if cnt > 1 else int(time.time() * RATE_HZ + i)

        if key not in trajectory:
            trajectory[key] = (arr[i], (stateTimestamp, i))
            new_traj += 1
        else:
            # trajectory[key] = (arr[i], (stateTimestamp, i))
            old_vec, _ = trajectory[key]

            # ---------- 高斯权重 ----------
            dist  = (L - 1 - i)                  # 与段尾的距离
            w_new = math.exp(-(dist ** 2) / (2 * sigma ** 2))
            w_old = 1.0 - w_new                  # 保证两权重和为 1
            # --------------------------------

            fused = old_vec * w_old + arr[i] * w_new
            trajectory[key] = (fused, (stateTimestamp, i))

    
    print(f"Length of trajectory: {len(trajectory)}, new points: {new_traj}")

class ExecutionNode:
    def __init__(self):
        global callback
        rospy.init_node('inference_node', anonymous=True)
        self.pub_arm0 = rospy.Publisher('/arm0/joint_states', JointState, queue_size=1)
        self.pub_arm1 = rospy.Publisher('/arm1/joint_states', JointState, queue_size=1)
        self.main_index = 0  # Initialize the action index
        self.subscriber = rospy.Subscriber('/predicted_trajectory', Float64MultiArray, callback)
        rospy.loginfo("Execution node initialized successfully.")

    def reset_env(self):
        """Reset both arms to their default positions and stop their movement."""
        rospy.loginfo("Resetting arms to initial position.")

        # Reset arm0
        joint_state_arm0 = JointState()
        joint_state_arm0.header.seq = self.main_index
        joint_state_arm0.header.stamp = rospy.Time.now()
        joint_state_arm0.header.frame_id = ""
        joint_state_arm0.name = [f"joint_{i}" for i in range(JOINT_COUNT)]
        joint_state_arm0.position = RESET_POSITION  # Reset positions
        joint_state_arm0.velocity = RESET_VELOCITY  # Set velocity to zero
        joint_state_arm0.effort = RESET_EFFORT  # Set effort to zero
        self.pub_arm0.publish(joint_state_arm0)

        # Reset arm1
        joint_state_arm1 = JointState()
        joint_state_arm1.header.seq = self.main_index
        joint_state_arm1.header.stamp = rospy.Time.now()
        joint_state_arm1.header.frame_id = ""
        joint_state_arm1.name = [f"joint_{i + JOINT_COUNT}" for i in range(JOINT_COUNT)]
        joint_state_arm1.position = RESET_POSITION  # Reset positions
        joint_state_arm1.velocity = RESET_VELOCITY  # Set velocity to zero
        joint_state_arm1.effort = RESET_EFFORT  # Set effort to zero
        self.pub_arm1.publish(joint_state_arm1)

        # rospy.loginfo("Arms reset successfully to position 0.")

    def step(self, action , seq: int):
        state_len = int(len(action) / 2)

        arm0 = action[:state_len].copy()
        gripper_state_arm0 = arm0[-1]
        arm0[-1] = (1.0 - gripper_state_arm0) * 0.08
        if arm0[-1] < 0.04:
            arm0[-1] = 0.0
        else: 
            arm0[-1] = 0.08

        arm1 = action[state_len:].copy()
        gripper_state_arm1 = arm1[-1]
        arm1[-1] = (1.0 - gripper_state_arm1) * 0.08
        if arm1[-1] < 0.04:
            arm1[-1] = 0.0
        else: 
            arm1[-1] = 0.08

        joint_state_arm0 = JointState()
        joint_state_arm0.header.seq = seq
        joint_state_arm0.header.stamp = rospy.Time.now()
        joint_state_arm0.name = [f"joint_{i}" for i in range(7)]
        joint_state_arm0.position = arm0
        joint_state_arm0.velocity = [10.0] * 7
        joint_state_arm0.effort = [20.0] * 7
        self.pub_arm0.publish(joint_state_arm0)

        joint_state_arm1 = JointState()
        joint_state_arm1.header.seq = seq
        joint_state_arm1.header.stamp = rospy.Time.now()
        joint_state_arm1.name = [f"joint_{i+7}" for i in range(7)]
        joint_state_arm1.position = arm1
        joint_state_arm1.velocity = [10.0] * 7
        joint_state_arm1.effort = [20.0] * 7
        self.pub_arm1.publish(joint_state_arm1)
        # rospy.loginfo(f"Published action {seq}: arm0={arm0[:3]}..., arm1={arm1[:3]}...")


def main():
    global exec_idx
    node = ExecutionNode()
    rate = rospy.Rate(RATE_HZ)
    key=None;
    start_pointer = 0

    while not rospy.is_shutdown():
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        if key is None and trajectory:
            key=sorted(trajectory.keys())[0];
            start_pointer = key
        
        if key in trajectory:
            print("################## EXEC: ", time.time(), key, trajectory[key][1])
            print("################## EXEC: ", key - start_pointer)
            node.step(trajectory[key][0].tolist(), key);
            key+=1;

        else:
            print("################## EXEC: null", time.time());
        rate.sleep()





if __name__ == '__main__':
    main()

