import cv2
import os

# 定义保存图像的目录
left_dir = './data/left'
right_dir = './data/right'

# 打开视频捕捉设备
cap = cv2.VideoCapture('/dev/video2')

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 设置高度

# 检查设备是否成功打开
if not cap.isOpened():
    print("无法打开视频捕捉设备")
    exit()

effect_cnt = 0

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧")
        break

    # 进行图像裁剪
    left_image = frame[:, :1280]    # 左半部分
    right_image = frame[:, 1280:]   # 右半部分

    # 进行棋盘格检查 棋盘格大小为 11 x 8
    chessboard_size = (11, 8)  # 棋盘格内角点的行列数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # 彩色图转灰度图
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # 找到棋盘格角点
    found_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size)
    found_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size)

    # 如果找到角点，进行亚像素精确化
    if found_left & found_right:
        cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray_left, chessboard_size, corners_left, found_left)

        cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray_right, chessboard_size, corners_right, found_right)

        cv2.imshow('Left Image', gray_left)
        cv2.imshow('Right Image', gray_right)
    else:
        cv2.imshow('Left Image', left_image)
        cv2.imshow('Right Image', right_image)

    # 按 's' 键保存左图和右图
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # 检查并创建目录
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        cv2.imwrite(f'{left_dir}/{effect_cnt}.png', left_image)
        cv2.imwrite(f'{right_dir}/{effect_cnt}.png', right_image)
        print(f"左图已保存为 {left_dir}/{effect_cnt}.png")
        print(f"右图已保存为 {right_dir}/{effect_cnt}.png")

        effect_cnt += 1

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
