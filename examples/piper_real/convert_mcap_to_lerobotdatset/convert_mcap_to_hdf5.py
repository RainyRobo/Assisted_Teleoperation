import sys
import os
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import cv2
from mcap.reader import make_reader
import bisect
import h5py

def save_rgb_image(rgb, filename="decoded_image.jpg"):
    # Save the decoded image as a .jpg file
    cv2.imwrite(filename, rgb)
    print(f"Image saved as {filename}")

# img 从 buff 转换到 bgr 格式
def decode_compressed_image(msg):
    # The 'data' field contains the raw byte data of the image
    np_arr = np.frombuffer(msg.data, np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # [H,W,C]=[480，640，3] + BGR
                                                    # cv2.imwrite("img.jpg", img)，要求 img 是 bgr
    if image is None:
        print("Failed to decode image")
        return None
    return image



def images_encoding(imgs):
    encode_data = []
    max_len = 0

    for i, img in enumerate(imgs):
        if img is None:
            continue
        success, encoded_image = cv2.imencode('.jpg', img)  # Encode as JPEG
        if not success:
            continue
        jpeg_data = encoded_image.tobytes()
        encode_data.append(np.frombuffer(jpeg_data, dtype=np.uint8))
        max_len = max(max_len, len(jpeg_data))
        # print(f"图像 {i + 1} 编码后的数据长度：{len(jpeg_data)}")

    # Pad data to max_len
    padded_data = [
        np.pad(data, (0, max_len - len(data)), mode='constant', constant_values=0)
        for data in encode_data
    ]


    return padded_data, max_len


def data_transform(path, episode_num, save_path):
    begin = 0
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.mcap')]
    if not files:
        raise FileNotFoundError(f"No .mcap files found in {path}")
    assert episode_num <= len(files), f"Requested {episode_num} episodes, but only {len(files)} .mcap files found"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(0, episode_num):
        mcap_file_path = os.path.join(path, files[i])  # 使用实际文件名
        print(f"Processing file: {mcap_file_path}")
        with open(mcap_file_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            # 数据存储
            camera_high = []
            camera_low = []
            camera_left_wrist = []
            camera_right_wrist = []
            state = []
            action_buffer = []
            gripper_buffer =[]
            action = []
            state_matched = []
            action_matched = []

            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                topic = channel.topic

                # cam_high = [] 高位相机数据
                if topic.startswith("/ob_camera_03/color/image_raw/compressed"):  #接收图像topic

                    # Debug1 获取图像编码信息 (JPEG)
                    # format = ros_msg.format
                    # print(f"Image format: {format}")

                    # Debug2: 消息的详细信息
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]: {ros_msg}")
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]")

                    # Use OpenCV Decoder JPEG to BGR format
                    bgr = decode_compressed_image(ros_msg)
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = bgr[:, :, ::-1]  # 通过在通道上进行第三维度反转从 BGR 转换为 RGB
                    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    camera_high.append({"log_time": message.log_time, "msg": rgb})



                # cam_low = [] 低位相机数据
                elif topic.startswith("/ob_camera_01/color/image_raw/compressed"):
                    bgr = decode_compressed_image(ros_msg)
                    rgb = bgr[:, :, ::-1]
                    camera_low.append({"log_time": message.log_time, "msg": rgb})




                # left_wrist = [] left_wrist相机数据
                elif topic.startswith("/ob_camera_04/color/image_raw/compressed"):
                    bgr = decode_compressed_image(ros_msg)
                    rgb = bgr[:, :, ::-1]
                    camera_left_wrist.append({"log_time": message.log_time, "msg": rgb})



                # right_wrist camera
                elif topic.startswith("/ob_camera_02/color/image_raw/compressed"):
                    bgr = decode_compressed_image(ros_msg)
                    rgb = bgr[:, :, ::-1]
                    camera_right_wrist.append({"log_time": message.log_time, "msg": rgb})

                # 状态
                elif topic.startswith("io_teleop/joint_states"):
                    # Debug1:打印解析出的消息内容
                    # print("Received JointState message:")
                    # print(f"Position: {ros_msg.position}")
                    # print(f"Velocity: {ros_msg.velocity}")
                    # print(f"Effort: {ros_msg.effort}")
                    # print(f"Joint Names: {ros_msg.name}")

                    right_state = list(ros_msg.position[:6])  # 右臂 6 关节角
                    right_state.append(ros_msg.position[6] * 2 / 0.08)  # 右臂夹爪归一化
                    # print(right_state)

                    left_state = list(ros_msg.position[8:14])
                    left_state.append(ros_msg.position[14] * 2 / 0.08)
                    # print(left_state)
                    state.append({"log_time": message.log_time, "msg": right_state+left_state})


                # action
                elif topic.startswith("io_teleop/joint_cmd"):
                    # Debug1:打印解析出的消息内容
                    # print("Received JointState message:")
                    # print(f"Position: {ros_msg.position}")
                    # print(f"Velocity: {ros_msg.velocity}")
                    # print(f"Effort: {ros_msg.effort}")
                    # print(f"Joint Names: {ros_msg.name}")

                    left_joint_cmd = list(ros_msg.position[:6])
                    right_joint_cmd = list(ros_msg.position[6:])
                    action_buffer.append({
                        "log_time": message.log_time,
                        "left_joint_cmd": left_joint_cmd,
                        "right_joint_cmd": right_joint_cmd,
                        "left_gripper": None,  # 待填充
                        "right_gripper": None  # 待填充
                    })

                # gripper
                elif topic.startswith("io_teleop/target_gripper_status"):
                    # Debug1:打印解析出的消息内容
                    # print("Received JointState message:")
                    # print(f"Position: {ros_msg.position}")
                    # print(f"Velocity: {ros_msg.velocity}")
                    # print(f"Effort: {ros_msg.effort}")
                    # print(f"Joint Names: {ros_msg.name}")


                    right_gripper = ros_msg.position[0]
                    left_gripper = ros_msg.position[1]
                    gripper_buffer.append({
                        "log_time": message.log_time,
                        "right_gripper": right_gripper,
                        "left_gripper": left_gripper
                    })

            # align action
            for action_idx in action_buffer:
                action_time = action_idx["log_time"]

                # 使用 bisect 找到最接近的时间戳
                gripper_times = [entry["log_time"] for entry in gripper_buffer]
                idx = bisect.bisect_left(gripper_times, action_time)

                # 找到左右两个最近的时间戳
                closest_gripper = None
                min_diff = float("inf")

                # 检查右侧（如果 idx 有效）
                if idx < len(gripper_buffer):
                    diff = abs(gripper_buffer[idx]["log_time"] - action_time)
                    if diff < min_diff:
                        min_diff = diff
                        closest_gripper = gripper_buffer[idx]

                # 检查左侧（如果 idx > 0）
                if idx > 0:
                    diff = abs(gripper_buffer[idx - 1]["log_time"] - action_time)
                    if diff < min_diff:
                        min_diff = diff
                        closest_gripper = gripper_buffer[idx - 1]

                # 生成对齐后的 action 数据
                if closest_gripper:
                    action.append({
                        "action_log_time": action_idx["log_time"],
                        "action_msg": [
                            *action_idx["right_joint_cmd"],  # First 6 elements: right_joint_cmd
                            closest_gripper["right_gripper"],  # 7th element: right_gripper
                            *action_idx["left_joint_cmd"],  # Next 6 elements: left_joint_cmd
                            closest_gripper["left_gripper"]  # 14th element: left_gripper
                        ]
                    })
                    # print(action)

            # 开始转换con
            # print(len(camera_high))
            # print(len(camera_low))
            # print(len(camera_left_wrist))
            # print(len(camera_right_wrist))


            min_length = min(len(camera_high), len(camera_low), len(camera_left_wrist),len(camera_right_wrist))
            # # print(min_length)
            fixed_frame_count = 300
            if min_length < fixed_frame_count:
                print(f"Episode {i} has only {min_length} frames, less than required {fixed_frame_count}. Skipping.")
                continue

            camera_high = camera_high[:fixed_frame_count]
            camera_low = camera_low[:fixed_frame_count]
            camera_left_wrist = camera_left_wrist[:fixed_frame_count]
            camera_right_wrist = camera_right_wrist[:fixed_frame_count]


            for j in range(len(camera_high)):
                camera_high_entry = camera_high[j]
                camera_low_entry = camera_low[j]
                camera_left_wrist_entry = camera_left_wrist[j]
                camera_right_wrist_entry = camera_right_wrist[j]

                # 例如，打印出当前帧的数据，或者进行进一步的处理
                # print(f"Frame {j}:")
                # print(f"Camera High: {camera_high_entry}")
                # print(f"Camera Low: {camera_low_entry}")
                # print(f"Left Wrist: {camera_left_wrist_entry}")
                # print(f"Right Wrist: {camera_right_wrist_entry}")

                camera_time = camera_high_entry["log_time"]

                # 使用 bisect 来找到最接近的时间戳
                # 找到在 sorted_state 中第一个大于或等于 camera_time 的索引
                idx = bisect.bisect_left([entry["log_time"] for entry in state], camera_time)

                # 找到左右两个最近的时间戳

                closest_state = None
                min_diff = float('inf')  # 正无穷大

                # 如果 idx 指向有效位置
                if idx < len(state):
                    diff = abs(state[idx]["log_time"] - camera_time)
                    if diff < min_diff:
                        closest_state = state[idx]
                        min_diff = diff

                # 如果 idx > 0，检查左侧的元素
                if idx > 0:
                    diff = abs(state[idx - 1]["log_time"] - camera_time)
                    if diff < min_diff:
                        closest_state = state[idx - 1]
                        min_diff = diff

                # 将找到的最接近的 state 元素存储到 state_matched 列表
                if closest_state:
                    state_matched.append({
                        "camera_log_time": camera_time,
                        "state_log_time": closest_state["log_time"],
                        "state_msg": closest_state["msg"],
                        "camera_high_msg": camera_high_entry["msg"],
                        "camera_low_msg": camera_low_entry["msg"],
                        "camera_left_wrist_msg": camera_left_wrist_entry["msg"],
                        "camera_right_wrist_msg": camera_right_wrist_entry["msg"]
                    })
                    # print(f"Aligned state frame {j}: camera={camera_time}, state={closest_state['log_time']} (diff: {min_diff})")

            # 对齐 action
            for j in range(len(camera_high)):
                camera_time = camera_high[j]["log_time"]
                idx = bisect.bisect_left([entry["action_log_time"] for entry in action], camera_time)
                closest_action = None
                action_diff = float('inf')

                if idx < len(action):
                    diff = abs(action[idx]["action_log_time"] - camera_time)
                    if diff < action_diff:
                        closest_action = action[idx]
                        action_diff = diff

                if idx > 0:
                    diff = abs(action[idx - 1]["action_log_time"] - camera_time)
                    if diff < action_diff:
                        closest_action = action[idx - 1]
                        action_diff = diff

                if closest_action and action_diff:
                    action_matched.append({
                        "action_log_time": closest_action["action_log_time"],
                        "action_msg": closest_action["action_msg"]
                    })
                    # print(
                    #     f"Aligned action frame {j}: camera={camera_time}, action={closest_action['action_log_time']} (diff: {action_diff})")




            qpos = []
            actions = []
            cam_high = []
            cam_low = []
            cam_left_wrist = []
            cam_right_wrist = []

            # 确保 state_matched 和 action_matched 长度一致
            min_matched_length = min(len(state_matched), len(action_matched))
            state_matched = state_matched[:min_matched_length]
            action_matched = action_matched[:min_matched_length]

            for j in range(min_matched_length):
                qpos.append(np.array(state_matched[j]["state_msg"], dtype=np.float32))
                actions.append(np.array(action_matched[j]["action_msg"], dtype=np.float32))
                cam_high.append(state_matched[j]["camera_high_msg"])
                cam_low.append(state_matched[j]["camera_low_msg"])
                cam_left_wrist.append(state_matched[j]["camera_left_wrist_msg"])
                cam_right_wrist.append(state_matched[j]["camera_right_wrist_msg"])

            # 保存到HDF5
            hdf5path = os.path.join(save_path, f'episode_{i}.hdf5')
            with h5py.File(hdf5path, 'w') as f:
                f.create_dataset('action', data=np.array(actions, dtype=np.float32))
                obs = f.create_group('observations')
                obs.create_dataset('qpos', data=np.array(qpos, dtype=np.float32))
                image = obs.create_group('images')

                # 编码并保存图像
                cam_high_enc, len_high = images_encoding(cam_high)
                cam_low_enc, len_low = images_encoding(cam_low)
                cam_left_enc, len_left = images_encoding(cam_left_wrist)
                cam_right_enc, len_right = images_encoding(cam_right_wrist)

                image.create_dataset('cam_high', data=np.array(cam_high_enc, dtype=np.uint8))
                image.create_dataset('cam_low', data=np.array(cam_low_enc, dtype=np.uint8))
                image.create_dataset('cam_left_wrist', data=np.array(cam_left_enc, dtype=np.uint8))
                image.create_dataset('cam_right_wrist', data=np.array(cam_right_enc, dtype=np.uint8))
        begin += 1
        print(f"Process {i} success!")
    return begin


'''
episode_i.hdf5:
    - action
    - observations
        - qpos
        - images
            - cam_high
            - cam_low
            - cam_left_wrist
            - cam_right_wrist
'''
if __name__ == "__main__":
    data_transform("/data/liuyu/fold_clothes_mcap", 25, "/home/research1/data/fold_clothes_hdf5")