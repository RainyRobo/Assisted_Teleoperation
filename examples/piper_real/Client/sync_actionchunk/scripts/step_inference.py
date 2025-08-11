#!/usr/bin/env python3
"""ROS node for publishing predicted trajectories based on camera and joint state inputs."""

import rospy
import numpy as np
from typing import Dict, Optional, Tuple, List
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge
import cv2
import time
import os
from openpi_client import image_tools, websocket_client_policy


class PI0:
    """Policy Inference Object for processing observations and generating actions."""

    def __init__(self, host: str = "192.168.2.250", port: int = 8000, img_size: Tuple[int, int] = (224, 224)):
        """Initialize the PI0 model with a WebSocket policy client.

        Args:
            host (str): WebSocket server host address.
            port (int): WebSocket server port.
            img_size (Tuple[int, int]): Target size for resizing images.
        """
        rospy.loginfo("Loading model...")
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.img_size = img_size
        self.observation_window: Optional[Dict] = None
        self.instruction: str = "uncap the pen"
        rospy.loginfo("Model loaded successfully!")

    def reset_observation_window(self) -> None:
        """Reset the observation window to an empty state."""
        self.observation_window = None

    def update_observation_window(
        self,
        state: np.ndarray,
        cam_high: np.ndarray,
        cam_low: np.ndarray,
        cam_left_wrist: np.ndarray,
        cam_right_wrist: np.ndarray
    ) -> None:
        """Update the observation window with state and camera images.

        Args:
            state (np.ndarray): Robot joint state.
            cam_high (np.ndarray): High camera image.
            cam_low (np.ndarray): Low camera image.
            cam_left_wrist (np.ndarray): Left wrist camera image.
            cam_right_wrist (np.ndarray): Right wrist camera image.
        """
        self.observation_window = {
            "state": np.array(state, dtype=np.float32),
            "images": {
                "cam_high": self._process_image(cam_high),
                "cam_low": self._process_image(cam_low),
                "cam_left_wrist": self._process_image(cam_left_wrist),
                "cam_right_wrist": self._process_image(cam_right_wrist)
            },
            "prompt": self.instruction,
        }
        rospy.loginfo(f"Observation window updated! Image shape: {self.observation_window['images']['cam_high'].shape}")

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """Process an image by resizing and converting to uint8.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            np.ndarray: Processed image in (C, H, W) format.

        Raises:
            ValueError: If the input image is invalid.
        """
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape if image is not None else None}")
        resized = image_tools.resize_with_pad(image, *self.img_size)
        return image_tools.convert_to_uint8(resized).transpose(2, 0, 1)

    def get_action(self) -> np.ndarray:
        """Infer actions from the observation window.

        Returns:
            np.ndarray: Predicted actions.

        Raises:
            ValueError: If observation window is not initialized or action shape is invalid.
        """
        if self.observation_window is None:
            raise ValueError("Observation window must be updated before getting actions.")
        try:
            actions = self.policy.infer(self.observation_window)["actions"]
            if actions.shape != (50, 14):
                raise ValueError(f"Expected action shape (50, 14), got {actions.shape}")
            return actions
        except Exception as e:
            rospy.logerr(f"Failed to infer actions: {e}")
            raise

    def save_debug_images(
        self,
        cam_high: np.ndarray,
        cam_low: np.ndarray,
        cam_left_wrist: np.ndarray,
        cam_right_wrist: np.ndarray
    ) -> None:
        """Save camera images as JPEG files for debugging.

        Args:
            cam_high (np.ndarray): High camera image.
            cam_low (np.ndarray): Low camera image.
            cam_left_wrist (np.ndarray): Left wrist camera image.
            cam_right_wrist (np.ndarray): Right wrist camera image.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.dirname(os.path.abspath(__file__))

        cameras = {
            "cam_high": cam_high,
            "cam_low": cam_low,
            "cam_left_wrist": cam_left_wrist,
            "cam_right_wrist": cam_right_wrist
        }

        for cam_name, cam_img in cameras.items():
            if cam_img is not None and len(cam_img.shape) == 3 and cam_img.shape[2] == 3:
                filename = os.path.join(debug_dir, f"{cam_name}_{timestamp}.jpg")
                cv2.imwrite(filename, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
                rospy.loginfo(f"Saved debug image: {filename}")
            else:
                rospy.logwarn(f"Skipping save for {cam_name}: Invalid image shape {cam_img.shape if cam_img is not None else None}")


class InferenceNode:
    """ROS node for subscribing to camera and joint state topics and publishing actions."""

    def __init__(self):
        """Initialize the InferenceNode with ROS subscribers and publishers."""
        # Load configuration from ROS parameters
        self.inference_loop = rospy.get_param("~inference_loop", 1)
        self.interpolate_step = rospy.get_param("~interpolate_step", 0)
        self.inference_step = rospy.get_param("~inference_step", 5)
        self.save_debug_images = rospy.get_param("~save_debug_images", False)
        self.max_steps = rospy.get_param("~max_steps", 1000)  # Maximum steps before reset
        self.reset_joint_angles = rospy.get_param("~reset_joint_angles", {
            "arm0": [0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.08],  # Same as arm1 for consistency
            "arm1": [0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.08]   # From provided rostopic command
        })
        self.reset_velocity = rospy.get_param("~reset_velocity", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # From provided command
        self.reset_effort = rospy.get_param("~reset_effort", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])    # From provided command

        rospy.init_node('inference_node', anonymous=True)
        self.bridge = CvBridge()

        # Image and joint state variables
        self.cam_high: Optional[np.ndarray] = None
        self.cam_low: Optional[np.ndarray] = None
        self.cam_left_wrist: Optional[np.ndarray] = None
        self.cam_right_wrist: Optional[np.ndarray] = None
        self.joint_left: Optional[List[float]] = None
        self.joint_right: Optional[List[float]] = None
        self.state: Optional[np.ndarray] = None
        self.action_count: int = 0  # Track number of executed actions

        # Publishers
        self.pub_arm0 = rospy.Publisher('/arm0/joint_states', JointState, queue_size=1)
        self.pub_arm1 = rospy.Publisher('/arm1/joint_states', JointState, queue_size=1)

        # Subscribers
        self.subscribers: List[rospy.Subscriber] = []
        self._setup_subscribers()

    def _setup_subscribers(self) -> None:
        """Set up ROS subscribers for camera images and joint states."""
        self.subscribers = [
            rospy.Subscriber("/ob_camera_01/color/image_raw", Image, self.cam_high_callback, queue_size=1, tcp_nodelay=True),
            rospy.Subscriber("/ob_camera_02/color/image_raw", Image, self.cam_low_callback, queue_size=1, tcp_nodelay=True),
            rospy.Subscriber("/ob_camera_03/color/image_raw", Image, self.cam_right_wrist_callback, queue_size=1, tcp_nodelay=True),
            rospy.Subscriber("/ob_camera_04/color/image_raw", Image, self.cam_left_wrist_callback, queue_size=1, tcp_nodelay=True),
            rospy.Subscriber("/arm0/joint_states_single", JointState, self.joint_right_callback, queue_size=1),
            rospy.Subscriber("/arm1/joint_states_single", JointState, self.joint_left_callback, queue_size=1)
        ]

    def cam_high_callback(self, msg: Image) -> None:
        """Callback for high camera image."""
        self.cam_high = self._convert_image(msg)

    def cam_low_callback(self, msg: Image) -> None:
        """Callback for low camera image."""
        self.cam_low = self._convert_image(msg)

    def cam_left_wrist_callback(self, msg: Image) -> None:
        """Callback for left wrist camera image."""
        self.cam_left_wrist = self._convert_image(msg)

    def cam_right_wrist_callback(self, msg: Image) -> None:
        """Callback for right wrist camera image."""
        self.cam_right_wrist = self._convert_image(msg)

    def _convert_image(self, msg: Image) -> Optional[np.ndarray]:
        """Convert ROS image message to OpenCV image.

        Args:
            msg (Image): ROS image message.

        Returns:
            Optional[np.ndarray]: OpenCV image in RGB format or None if conversion fails.
        """
        try:
            return self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return None

    def joint_left_callback(self, msg: JointState) -> None:
        """Callback for left arm joint states."""
        self.joint_left = self._process_joint_state(msg)

    def joint_right_callback(self, msg: JointState) -> None:
        """Callback for right arm joint states."""
        self.joint_right = self._process_joint_state(msg)

    def _process_joint_state(self, msg: JointState) -> List[float]:
        """Process joint state message, normalizing gripper state.

        Args:
            msg (JointState): ROS joint state message.

        Returns:
            List[float]: Processed joint positions.
        """
        positions = list(msg.position)
        if positions:
            gripper_state = positions[-1] / 0.079
            positions[-1] = 1.0 - gripper_state
        return positions

    def update_state(self) -> bool:
        """Update the combined robot state from joint states.

        Returns:
            bool: True if state is valid, False otherwise.
        """
        if self.joint_left is None or self.joint_right is None:
            return False
        if len(self.joint_left) == 7 and len(self.joint_right) == 7:
            self.state = np.concatenate((self.joint_right, self.joint_left))
            return True
        rospy.logwarn(f"Invalid joint state dimensions: left={len(self.joint_left)}, right={len(self.joint_right)}")
        return False

    def all_data_ready(self) -> bool:
        """Check if all required data is available.

        Returns:
            bool: True if all data is ready, False otherwise.
        """
        data = {
            "cam_high": self.cam_high,
            "cam_low": self.cam_low,
            "cam_left_wrist": self.cam_left_wrist,
            "cam_right_wrist": self.cam_right_wrist,
            "joint_left": self.joint_left,
            "joint_right": self.joint_right
        }
        missing = [key for key, value in data.items() if value is None]
        if missing:
            rospy.logwarn(f"Missing data: {', '.join(missing)}")
            return False
        return True

    def publish_action(self, action: np.ndarray, seq: int) -> None:
        """Publish actions for both arms as JointState messages.

        Args:
            action (np.ndarray): Action array with 14 degrees of freedom (7 per arm).
            seq (int): Sequence number for the message header.
        """
        # Right arm (first 7 DOF)
        arm0 = action[:, :7].copy()
        gripper_state_arm0 = arm0[:, -1]
        arm0[:, -1] = (1.0 - gripper_state_arm0) * 0.08

        # Left arm (last 7 DOF)
        arm1 = action[:, 7:].copy()
        gripper_state_arm1 = arm1[:, -1]
        arm1[:, -1] = (1.0 - gripper_state_arm1) * 0.08

        # Create and publish JointState for arm0
        joint_state_arm0 = JointState()
        joint_state_arm0.header.seq = seq
        joint_state_arm0.header.stamp = rospy.Time.now()
        joint_state_arm0.header.frame_id = ""
        joint_state_arm0.name = [f"joint_{i}" for i in range(7)]
        joint_state_arm0.position = arm0.flatten()
        joint_state_arm0.velocity = [10] * 7
        joint_state_arm0.effort = [10] * 7
        self.pub_arm0.publish(joint_state_arm0)

        # Create and publish JointState for arm1
        joint_state_arm1 = JointState()
        joint_state_arm1.header.seq = seq
        joint_state_arm1.header.stamp = rospy.Time.now()
        joint_state_arm1.header.frame_id = ""
        joint_state_arm1.name = [f"joint_{i+7}" for i in range(7)]
        joint_state_arm1.position = arm1.flatten()
        joint_state_arm1.velocity = [10] * 7
        joint_state_arm1.effort = [20] * 7
        self.pub_arm1.publish(joint_state_arm1)

        rospy.loginfo(f"Published action {seq}: arm0={arm0.flatten()[:3]}..., arm1={arm1.flatten()[:3]}...")

    def reset(self, seq: int) -> None:
        """Reset both arms to fixed joint angles, unsubscribe from all topics, and stop the node.

        Args:
            seq (int): Current sequence number.

        Raises:
            rospy.ROSInterruptException: To signal the node to stop after reset.
        """
        rospy.loginfo(f"Resetting arms to fixed joint angles: {self.reset_joint_angles}")

        # Publish reset state for arm0
        joint_state_arm0 = JointState()
        joint_state_arm0.header.seq = seq
        joint_state_arm0.header.stamp = rospy.Time.now()
        joint_state_arm0.header.frame_id = ""
        joint_state_arm0.name = [f"joint_{i}" for i in range(7)]
        joint_state_arm0.position = self.reset_joint_angles["arm0"]
        joint_state_arm0.velocity = self.reset_velocity
        joint_state_arm0.effort = self.reset_effort
        self.pub_arm0.publish(joint_state_arm0)

        # Publish reset state for arm1
        joint_state_arm1 = JointState()
        joint_state_arm1.header.seq = seq
        joint_state_arm1.header.stamp = rospy.Time.now()
        joint_state_arm1.header.frame_id = ""
        joint_state_arm1.name = [f"joint_{i+7}" for i in range(7)]
        joint_state_arm1.position = self.reset_joint_angles["arm1"]
        joint_state_arm1.velocity = self.reset_velocity
        joint_state_arm1.effort = self.reset_effort
        self.pub_arm1.publish(joint_state_arm1)

        # Unsubscribe from all topics
        for subscriber in self.subscribers:
            subscriber.unregister()
            rospy.loginfo(f"Unsubscribed from topic: {subscriber.name}")
        self.subscribers = []

        # Reset action count
        self.action_count = 0
        rospy.loginfo("Reset completed, stopping node.")
        raise rospy.ROSInterruptException("Node stopped after reset.")

    def run(self, model: PI0) -> None:
        """Run the inference loop, processing observations and publishing actions.

        Args:
            model (PI0): Policy inference model.
        """
        rate = rospy.Rate(self.inference_loop)
        seq = 0

        while not rospy.is_shutdown():
            if self.all_data_ready() and self.update_state():
                try:
                    model.update_observation_window(
                        self.state,
                        self.cam_high,
                        self.cam_low,
                        self.cam_left_wrist,
                        self.cam_right_wrist
                    )
                    if self.save_debug_images:
                        model.save_debug_images(self.cam_high, self.cam_low, self.cam_left_wrist, self.cam_right_wrist)

                    actions = model.get_action()
                    for i in range(min(20, actions.shape[0])):
                        if self.action_count >= self.max_steps:
                            self.reset(seq)
                            return  # Exit after reset (unreachable due to exception, but for clarity)

                        action = np.array(actions[i:i+1], copy=True)
                        self.publish_action(action, seq)
                        seq += 1
                        self.action_count += 1
                except rospy.ROSInterruptException:
                    rospy.loginfo("Node interrupted by reset or shutdown.")
                    return
                except Exception as e:
                    rospy.logerr(f"Error in inference loop: {e}")
            rate.sleep()


def interpolate(prev: np.ndarray, next: np.ndarray) -> np.ndarray:
    """Perform linear interpolation between two action arrays.

    Args:
        prev (np.ndarray): Previous action array.
        next (np.ndarray): Next action array.

        Returns:
            np.ndarray: Interpolated action array.

        Raises:
            ValueError: If input arrays have incompatible shapes.
    """
    if prev.shape != next.shape:
        raise ValueError(f"Incompatible shapes for interpolation: prev={prev.shape}, next={next.shape}")
    alpha = np.linspace(0, 1, len(prev)).reshape(-1, 1)
    return (1 - alpha) * prev + alpha * next


def main():
    """Main function to initialize and run the inference node."""
    try:
        # Load configuration from ROS parameters
        host = rospy.get_param("~policy_host", "192.168.2.250")
        port = rospy.get_param("~policy_port", 8000)
        img_size = rospy.get_param("~img_size", [224, 224])

        model = PI0(host=host, port=port, img_size=tuple(img_size))
        node = InferenceNode()
        node.run(model)
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted.")
    except Exception as e:
        rospy.logerr(f"Node failed: {e}")


if __name__ == '__main__':
    main()