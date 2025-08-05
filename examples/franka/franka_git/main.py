from openpi_client import image_tools
from openpi_client import websocket_client_policy
import panda_py
from panda_py import libfranka
import rospy
from sensor_msgs.msg import JointState, Image
import cv2
from cv_bridge import CvBridge
import numpy as np

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

class PI0:
    def __init__(self):
        self.policy = websocket_client_policy.WebsocketClientPolicy(host="192.168.2.250", port=8000)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = "Stack the cups in a counterclockwise direction."
        # self.instruction = "do nothing."

    def update_observation_window(self, img, wrist_img, joints, gripper):
        self.observation_window = {
            "joints": np.array(joints, dtype=np.float32),
            "gripper":  np.array(gripper, dtype=np.float32),
            "base_rgb": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            "wrist_rgb":  image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224)),
            "prompt": self.instruction,
        }

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]
    
class panda_robot:
    def __init__(self, hostname):
        self.panda = panda_py.Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self.panda.set_default_behavior() # Panda default collision thresholds

        self.img = None
        self.wrist_img = None
        self.bridge = CvBridge()
        self.joint = self.panda.get_state().q
        self.gripper_width = 1-self.gripper.read_once().width/0.08
        if self.gripper_width < 0.5:
            self.gripper_state = "open"
        else:
            self.gripper_state = "closed"

        self.joint_speed_factor = 0.25
        self.cart_speed_factor = 0.15
        self.stiffness = [120, 120, 120, 150, 80, 50, 15]  # 关节4-7刚度提升25-50%
        self.damping = [30, 30, 30, 20, 10, 6, 3]          # 阻尼同步优化
        self.dq_threshold = 0.003
        self.success_threshold = 0.03
        self.panda.get_robot().set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

        rospy.init_node("eval_pi0", anonymous=True)
        # 2个订阅者
        rospy.Subscriber("/ob_camera_01/color/image_raw", Image, self.img_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber("/ob_camera_02/color/image_raw", Image, self.wrist_img_callback, queue_size=1000, tcp_nodelay=True)
    
    def img_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.img = img

    def wrist_img_callback(self, msg):
        wrist_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self.wrist_img = wrist_img

    def move_joint(self, qs):
        self.panda.move_to_joint_position(qs,
                             speed_factor=self.joint_speed_factor,
                             stiffness=self.stiffness,
                             damping=self.damping,
                             dq_threshold=self.dq_threshold,
                             success_threshold=self.success_threshold)
        
    def action_gripper(self, width):
        # 首次触发逻辑
        if width >= 0.5 and self.gripper_state != "closed":
            self.gripper.move(0.00, 1.0)  # 关闭夹爪
            self.gripper_state = "closed"  # 更新状态
        elif width < 0.5 and self.gripper_state != "open":
            self.gripper.move(0.08, 1.0)  # 打开夹爪
            self.gripper_state = "open"    # 更新状态
    
    def get_joints(self):
        return self.panda.get_state().q
    
    def get_gripper(self):
        return 1-self.gripper.read_once().width/0.08
    
def main():
    model = PI0()
    robot = panda_robot('192.168.5.12')

    rate = rospy.Rate(15)

    while not rospy.is_shutdown():
        if robot.img is not None and robot.wrist_img is not None:
            robot.joint = robot.get_joints()
            robot.gripper_width = robot.get_gripper()
            model.update_observation_window(robot.img, robot.wrist_img, robot.joint , robot.gripper_width)
            actions = model.get_action()
            assert actions.shape == (50, 8)
            actions = actions[:12]
            for i in range(12):
                robot.move_joint(actions[i][:7])
                robot.action_gripper(actions[i][7])

            # qs = [actions[i][:7] for i in range(40)]
            # robot.move_joint(qs)
            # robot.action_gripper(actions[i][7])
            robot.img = None
            robot.wrist_img = None
        # rate.sleep()

if __name__ == '__main__':
    main()
