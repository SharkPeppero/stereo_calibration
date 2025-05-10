import sys
import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import pyzed.sl as sl

image_cnt = 0

"""
该脚本兼容多种双目相机采集方式（支持 RealSense、ZED 以及指定设备端口 other 模式），用于从双目相机中提取左右图像。
如果指定设备为 other，则需要额外通过 /dev/video 端口指定左右图像源（例如 /dev/video0 /dev/video2）。
配置参数支持：
    1. 分辨率
    2. 帧率
    3. 图像格式（默认 YUYV）
    4. 图像保存目录：自动在该目录下创建 left/ 与 right/ 子目录
"""

# ------------------- 解析命令行参数 -------------------
parser = argparse.ArgumentParser(description='Capture stereo images from various sources.')
parser.add_argument('--device', type=str, choices=['realsense', 'zed', 'other'], default='realsense',
                    help='Choose the stereo camera type')
parser.add_argument('--resolution', type=str, default='640x480',
                    help='Resolution of the grayscale images (e.g., 640x480)')
parser.add_argument('--fps', type=int, default=30, help='Frame rate (fps)')
parser.add_argument('--format', type=str, default='YUYV', help='Image format (default YUYV)')
parser.add_argument('--save_dir', type=str, default='./data/', help='Root directory to save stereo images')
parser.add_argument('--left_dev', type=str, help='Left camera device (e.g., /dev/video0) for other mode')
parser.add_argument('--right_dev', type=str, help='Right camera device (e.g., /dev/video2) for other mode')
parser.add_argument('--chessboard', type=str, default='11x8', help='Chessboard grid size, e.g., 11x8')
args = parser.parse_args()

width, height = map(int, args.resolution.split('x'))
fps = args.fps
chessboard_size = tuple(map(int, args.chessboard.split('x')))

# 按照日期创建保存目录
save_dir = os.path.join(args.save_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(save_dir, exist_ok=True)
left_path = os.path.join(save_dir, 'left')
right_path = os.path.join(save_dir, 'right')
os.makedirs(left_path, exist_ok=True)
os.makedirs(right_path, exist_ok=True)

print(f"Device: {args.device}\nResolution: {width}x{height}\nFPS: {fps}\nFormat: {args.format}")

# ------------------- 图像显示函数 -------------------
def show_stereo_pair_with_chessboard(left_img, right_img):
    show_l = left_img.copy()
    show_r = right_img.copy()

    gray_l = cv2.cvtColor(show_l, cv2.COLOR_BGR2GRAY) if show_l.ndim == 3 else show_l
    gray_r = cv2.cvtColor(show_r, cv2.COLOR_BGR2GRAY) if show_r.ndim == 3 else show_r

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size)

    if ret_l:
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        cv2.drawChessboardCorners(show_l, chessboard_size, corners_l, ret_l)
    if ret_r:
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        cv2.drawChessboardCorners(show_r, chessboard_size, corners_r, ret_r)

    stacked = np.hstack((show_l, show_r))
    cv2.namedWindow("Stereo View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo View", 1280, 720)
    cv2.imshow("Stereo View", stacked)

# ------------------- RealSense 相机采集 -------------------
if args.device == 'realsense':
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)
            right_frame = frames.get_infrared_frame(2)

            if not left_frame or not right_frame:
                continue

            left_img = np.asanyarray(left_frame.get_data())
            right_img = np.asanyarray(right_frame.get_data())

            show_stereo_pair_with_chessboard(left_img, right_img)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(os.path.join(left_path, f'left_{image_cnt}.png'), left_img)
                cv2.imwrite(os.path.join(right_path, f'right_{image_cnt}.png'), right_img)
                print(f"Saved left: {left_path}/left_{image_cnt}.png")
                print(f"Saved right: {right_path}/right_{image_cnt}.png")
                image_cnt += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ------------------- ZED 相机采集 -------------------
elif args.device == 'zed':
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = fps

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] Failed to open ZED camera")
        sys.exit(1)

    left_image = sl.Mat()
    right_image = sl.Mat()

    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                left_np = left_image.get_data()
                right_np = right_image.get_data()

                show_stereo_pair_with_chessboard(left_np, right_np)

                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(os.path.join(left_path, f'left_{image_cnt}.png'), left_np)
                    cv2.imwrite(os.path.join(right_path, f'right_{image_cnt}.png'), right_np)
                    print(f"Saved left: {left_path}/left_{image_cnt}.png")
                    print(f"Saved right: {right_path}/right_{image_cnt}.png")
                    image_cnt += 1
    finally:
        zed.close()
        cv2.destroyAllWindows()

# ------------------- Other（指定 /dev/video）模式 -------------------
elif args.device == 'other':
    if not args.left_dev or not args.right_dev:
        print("[ERROR] --left_dev and --right_dev must be specified when using 'other' mode.")
        sys.exit(1)

    cap_left = cv2.VideoCapture(args.left_dev)
    cap_right = cv2.VideoCapture(args.right_dev)

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap_left.set(cv2.CAP_PROP_FPS, fps)

    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap_right.set(cv2.CAP_PROP_FPS, fps)

    try:
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()

            if not ret_l or not ret_r:
                print("[WARNING] Frame capture failed. Skipping...")
                continue

            show_stereo_pair_with_chessboard(frame_l, frame_r)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(os.path.join(left_path, f'left_{image_cnt}.png'), frame_l)
                cv2.imwrite(os.path.join(right_path, f'right_{image_cnt}.png'), frame_r)
                print(f"Saved left: {left_path}/left_{image_cnt}.png")
                print(f"Saved right: {right_path}/right_{image_cnt}.png")
                image_cnt += 1
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

else:
    print("[ERROR] Unknown device type.")
    sys.exit(1)