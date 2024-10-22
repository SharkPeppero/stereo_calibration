import sys

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os

image_cnt = 0

# 解析命令行参数
parser = argparse.ArgumentParser(description='Capture infrared images from D435i.')
parser.add_argument('--resolution', type=str, default='640x480', help='Resolution of the grayscale images (e.g., 640x480)')
parser.add_argument('--fps', type=int, default=30, help='FPS of the grayscale images')
parser.add_argument('--format', type=str, choices=['Y8', 'Y16'], default='Y8', help='Format of the grayscale images (Y8 or Y16)')
parser.add_argument('--left_path', type=str, default='./data/left/', help='Path to save left grayscale image')
parser.add_argument('--right_path', type=str, default='./data/right/', help='Path to save right grayscale image')
args = parser.parse_args()

# 解析分辨率
gray_width, gray_height = map(int, args.resolution.split('x'))
gray_fps = args.fps
gray_format = args.format

# 打印参数
print('gray_width: %s' % gray_width)
print('gray_height: %s' % gray_height)
print('gray_fps: %s' % gray_fps)
print('gray_format: %s' % gray_format)

# 创建管道
pipeline = rs.pipeline()

# 配置流
config = rs.config()

# 启用深度、彩色和红外流，并设置分辨率和帧率
if gray_format == 'Y8':
    config.enable_stream(rs.stream.depth, gray_width, gray_height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, gray_width, gray_height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.infrared, 1, gray_width, gray_height, rs.format.y8, gray_fps)  # 左侧红外流
    config.enable_stream(rs.stream.infrared, 2, gray_width, gray_height, rs.format.y8, gray_fps)  # 右侧红外流
elif gray_format == 'Y16':
    config.enable_stream(rs.stream.depth, gray_width, gray_height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, gray_width, gray_height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.infrared, 1, gray_width, gray_height, rs.format.y16, args.fps)  # 左侧红外流
    config.enable_stream(rs.stream.infrared, 2, gray_width, gray_height, rs.format.y16, args.fps)  # 右侧红外流

# 启动管道
pipeline.start(config)

# 获取流信息
profile = pipeline.get_active_profile()
left_ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
right_ir_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

# # 关闭 IR 发射器
try:
    device = profile.get_device()
    depth_sensor = device.query_sensors()[0]
    emitter = depth_sensor.get_option(rs.option.emitter_enabled)
    print("emitter = ", emitter)
    set_emitter = 0
    depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
    emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
    print("new emitter = ", emitter1)
except Exception as e:
    print(f"Failed to turn off IR emitters: {e}")
    sys.exit(1)

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()

        # 获取左侧和右侧红外图像
        left_ir = frames.get_infrared_frame(1)
        right_ir = frames.get_infrared_frame(2)

        # 检查图像是否可用
        if not left_ir or not right_ir:
            continue

        # 转成cv的图像格式
        left_image = np.asanyarray(left_ir.get_data())
        right_image = np.asanyarray(right_ir.get_data())



        # 获取分辨率、fps和格式
        width = left_ir.get_width()
        height = left_ir.get_height()
        fps = left_ir_profile.fps
        format = 'Y16' if left_ir_profile.format() == rs.format.y16 else 'Y8'

        # 进行棋盘格检查 棋盘格大小为 11 x 8
        chessboard_size = (11, 8)  # 棋盘格内角点的行列数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

        # 彩色图转灰度图
        # gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        # gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # 找到棋盘格角点
        found_left, corners_left = cv2.findChessboardCorners(left_image, chessboard_size)
        found_right, corners_right = cv2.findChessboardCorners(right_image, chessboard_size)

        # 如果找到角点，进行亚像素精确化
        if found_left & found_right:
            cv2.cornerSubPix(left_image, corners_left, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(left_image, chessboard_size, corners_left, found_left)

            cv2.cornerSubPix(right_image, corners_right, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(right_image, chessboard_size, corners_right, found_right)

            cv2.imshow('Left Image', left_image)
            cv2.imshow('Right Image', right_image)
        else:
            cv2.imshow('Left Image', left_image)
            cv2.imshow('Right Image', right_image)

        # # 显示图像
        # cv2.imshow('Left IR', left_image)
        # cv2.imshow('Right IR', right_image)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # 按下ESC键或'q'键退出
            break
        elif key == ord('s'):  # 按下's'键保存图像

            # 创建保存路径（如果不存在则创建）
            os.makedirs(args.left_path, exist_ok=True)
            os.makedirs(args.right_path, exist_ok=True)

            # 保存图像
            if format == 'Y8':
                cv2.imwrite(os.path.join(args.left_path, f'left_image_{image_cnt}.png'), left_image.astype(np.uint8))
                cv2.imwrite(os.path.join(args.right_path, f'right_image_{image_cnt}.png'), right_image.astype(np.uint8))
            else:
                cv2.imwrite(os.path.join(args.left_path, f'left_image_{image_cnt}.png'), left_image.astype(np.uint16))
                cv2.imwrite(os.path.join(args.right_path, f'right_image_{image_cnt}.png'), right_image.astype(np.uint16))

            # 打印保存的路径
            print('left—— image path: ', os.path.join(args.left_path, f'left_image_{image_cnt}.png'))
            print('left image path: ', os.path.join(args.right_path, f'right_image_{image_cnt}.png'))
            print()

            # 增加计数
            image_cnt += 1

finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()
