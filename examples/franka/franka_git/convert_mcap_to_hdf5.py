import sys
import os 
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import cv2
from mcap.reader import make_reader
import bisect
import h5py

# img 从 buff 转换到 bgr 格式
def decode_compressed_image(msg):
    # The 'data' field contains the raw byte data of the image
    np_arr = np.frombuffer(msg.data, np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # [H,W,C]=[480，640，3] + BGR
                                                    # cv2.imwrite("img.jpg", img)，要求 img 是 bgr
    if image is None:
        print("Failed to decode image")
    return image

# def save_rgb_image(rgb, filename="decoded_image.jpg"):
#     # Save the decoded image as a .jpg file
#     cv2.imwrite(filename, rgb)
#     print(f"Image saved as {filename}")

# def images_encoding(imgs):
#     encode_data = [] #这是一个空列表，用于存储每个图像编码后的数据。
#     padded_data = [] #这是一个空列表，用于存储经过填充后的编码数据
#
#     max_len = 0 #用于跟踪编码后的图像数据的最大长度，初始值为0。
#
#     for i in range(len(imgs)):
#         success, encoded_image = cv2.imencode('.jpg', imgs[i])  # 已经转成了 RGB, 因此编码保持 RGB
#         jpeg_data = encoded_image.tobytes()
#         encode_data.append(jpeg_data)
#         max_len = max(max_len, len(jpeg_data))
#     # padding
#     for i in range(len(imgs)):
#         padded_data.append(encode_data[i].ljust(max_len, b'\0'))
#     return encode_data, max_len

import cv2


def images_encoding(imgs):
    encode_data = []  # 这是一个空列表，用于存储每个图像编码后的数据。
    padded_data = []  # 这是一个空列表，用于存储经过填充后的编码数据。

    max_len = 0  # 用于跟踪编码后的图像数据的最大长度，初始值为0。

    print("开始处理图像编码...")  # 调试：开始处理
    for i in range(len(imgs)):
        print(f"正在编码第 {i + 1} 张图像...")  # 调试：显示当前处理的图像
        success, encoded_image = cv2.imencode('.jpg', imgs[i])  # 已经转成了 RGB, 因此编码保持 RGB

        if not success:
            print(f"图像 {i + 1} 编码失败！")
            continue  # 如果编码失败，跳过当前图片

        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)

        print(f"图像 {i + 1} 编码后的数据长度：{len(jpeg_data)}")  # 调试：输出每张图片的编码数据长度

        max_len = max(max_len, len(jpeg_data))  # 更新最大长度
        print(f"更新后的最大编码长度：{max_len}")  # 调试：显示更新后的最大长度

    # padding
    print("开始填充编码数据...")  # 调试：开始填充数据
    for i in range(len(imgs)):
        # 填充图像数据到最大长度
        padded_data.append(encode_data[i].ljust(max_len, b'\0'))
        print(f"图像 {i + 1} 填充后的数据长度：{len(padded_data[-1])}")  # 调试：显示填充后的数据长度

    print("图像编码和填充完成。")  # 调试：结束处理
    return encode_data, max_len


def data_transform(path, episode_num, save_path):
    begin = 0
    # 提取目标文件夹下的所有文件列表
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    assert episode_num <= len(files), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(0, episode_num):
        # mcap_file_path = os.path.join(path, f'episode_{i}.mcap')
        mcap_file_path = os.path.join(path, '20250112_DualPiper_pulling_01.mcap')
        with open(mcap_file_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            # channel.topic 话题名称
            # schema.name 消息类型
            # message.log_time 消息的记录时间
            # ros_msg 消息内容
            
            state = []
            camera_high = []
            camera_left = []

            state_matched = []
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                topic = channel.topic # 确定topic开始分发

                #
                if topic.startswith("/ob_camera_01/color/image_raw/compressed"):
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]: {ros_msg}")
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]")
                    bgr = decode_compressed_image(ros_msg)
                    rgb = bgr[:, :, ::-1]   # 保存成了 rgb
                    # save_rgb_image(rgb, "/home/io/ld/RoboTwin/decoded_image1.jpg")
                    camera_high.append({"log_time":message.log_time, "msg":rgb})


                elif topic.startswith("/ob_camera_01/color/image_raw/compressed"):
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]: {ros_msg}")
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]")
                    bgr = decode_compressed_image(ros_msg)
                    rgb = bgr[:, :, ::-1]
                    camera_left.append({"log_time":message.log_time, "msg":rgb})

                elif topic.startswith("io_teleop/joint_states"):
                    # print(f"{channel.topic} {schema.name} [{message.log_time}]: {ros_msg}")
                    #print(f"{channel.topic} {schema.name} [{message.log_time}]")
                    position_data = ros_msg.position[:7]
                    position_data.append(ros_msg.position[7] * 2 / 0.08)   # 0~1 的夹爪宽度
                    state.append({"log_time":message.log_time, "msg":position_data})

            min_length = min(len(camera_high), len(camera_left))
            if len(camera_high) != len(camera_left):
                camera_high = camera_high[:min_length]
                camera_left = camera_left[:min_length]

            for j in range(min_length):
                camera_high_entry = camera_high[j]
                camera_left_entry = camera_left[j]

                camera_time = camera_high_entry["log_time"]
                
                # 使用 bisect 来找到最接近的时间戳
                # 找到在 sorted_state 中第一个大于或等于 camera_time 的索引
                idx = bisect.bisect_left([entry["log_time"] for entry in state], camera_time)
                
                # 找到左右两个最近的时间戳

                closest_state = None
                min_diff = float('inf')     # 正无穷大
                
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
                        "camera_left_msg": camera_left_entry["msg"]
                    })
                
            qpos = []
            actions = []
            cam_high = []
            cam_left_wrist = []

            for entry in state_matched:
                # 提取 state_msg
                state_msg = np.array(entry["state_msg"])   # numpy 格式
                state_msg = state_msg.astype(np.float32)   # 转为float32类型
                # 将整个 state_msg 填充到 qpos 和 actions 中
                qpos.append(state_msg)  # 将 state_msg 填充到 qpos
                actions.append(state_msg)  # 将 state_msg 填充到 actions
                    
                # 填充 camera_high_msg 和 camera_left_msg
                cam_high.append(entry["camera_high_msg"])  # 将 camera_high_msg 填充到 cam_high
                cam_left_wrist.append(entry["camera_left_msg"])  # 将 camera_left_msg 填充到 cam_left_wrist
                
            hdf5path = os.path.join(save_path, f'episode_{i}.hdf5')
            with h5py.File(hdf5path, 'w') as f:
                f.create_dataset('action', data=np.array(actions))
                obs = f.create_group('observations')
                obs.create_dataset('qpos', data=np.array(qpos))
                image = obs.create_group('images')
                cam_high_enc, len_high = images_encoding(cam_high)          # rgb 图像重新压缩 [H,W,C]=[480，640，3] + RGB
                # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)

                # 定长存储
                # image.create_dataset('cam_high', data=cam_high_enc, dtype=f'S{len_high}') # 定长存储
                image.create_dataset('cam_high', data=cam_high_enc, dtype=h5py.special_dtype(vlen=bytes)) # 变长存储


                # image.create_dataset('cam_left', data=cam_left_wrist_enc, dtype=f'S{len_left}')
        begin += 1
        print(f"proccess {i} success!")
    return begin

'''
episode_i.hdf5:
    -action
    -observations
        -qpos(state)
        -images
            -cam_high
            -cam_left
'''
if  __name__ == "__main__":
    # data_transform("/home/io/ld/data/franka_stack_cup_0619", 294, "/home/io/ld/data/hdf5_0619")

    data_transform("/home/research1/data/piper_mcap", 1, "/home/research1/data/piper_hdf5")