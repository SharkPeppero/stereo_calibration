import os
import numpy as np
import cv2
import glob
import argparse

import json
import pickle
import yaml


class Stereo_Camera_Calibration(object):

    ### 初始化参数
    def __init__(self, width, height, lattice):
        self.width = width  # 棋盘格宽方向黑白格子相交点个数
        self.height = height  # 棋盘格长方向黑白格子相交点个数
        self.lattice = lattice  # 棋盘格每个格子的边长

        # 设置迭代终止条件
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)         # 单目标定的最大迭代次数以及迭代阈值
        self.criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)  # 双目标定的最大迭代参数以及迭代阈值

    ### 双目标定
    def stereo_calibration(self, left_files, right_files):

        # 棋盘格3D点
        objp = np.zeros((self.width * self.height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        objp *= self.lattice

        objpoints = []  # 3d points in real world space
        imgpointsR = []  # 2d points in image plane
        imgpointsL = []

        for i in range(len(left_files)):

            # 读取图片并进行二值化
            ChessImaL = cv2.imread(left_files[i], 0)
            ChessImaR = cv2.imread(right_files[i], 0)

            # 设置阈值
            _, ChessImaL_bin = cv2.threshold(ChessImaL, 50, 255, cv2.THRESH_BINARY)
            _, ChessImaR_bin = cv2.threshold(ChessImaR, 50, 255, cv2.THRESH_BINARY)

            retL, cornersL = cv2.findChessboardCorners(ChessImaL, (self.width, self.height), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)  # 提取左图每一张图片的角点
            retR, cornersR = cv2.findChessboardCorners(ChessImaR, (self.width, self.height), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)  # 提取右图每一张图片的角点

            if (True == retR) & (True == retL):
                objpoints.append(objp)
                cv2.cornerSubPix(ChessImaL, cornersL, (10, 10), (-1, -1), self.criteria)  # 亚像素精确化，对粗提取的角点进行精确化
                cv2.cornerSubPix(ChessImaR, cornersR, (10, 10), (-1, -1), self.criteria)  # 亚像素精确化，对粗提取的角点进行精确化
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)

                # ret_l = cv2.drawChessboardCorners(ChessImaL, (self.width, self.height), cornersL, retL)
                # cv2.imshow(left_files[i], ChessImaL)
                #
                # ret_r = cv2.drawChessboardCorners(ChessImaR, (self.width, self.height), cornersR, retR)
                # cv2.imshow(right_files[i], ChessImaR)
                # cv2.waitKey()
                #
                # cv2.destroyAllWindows()
            else:
                print('not find chessboard corners!!')
                print('left image: ' + left_files[i])
                print('right image: ' + right_files[i])

        # 相机的单双目标定、及校正
        #   左侧相机单独标定
        retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)
        #   右侧相机单独标定
        retR, K2, D2, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)

        # --------- 双目相机的标定 ----------#
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC  # K和D个矩阵是固定的。这是默认标志。如果你校准好你的相机，只求解𝑅,𝑇,𝐸,𝐹。
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT  # 修复K矩阵中的参考点。
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS    # K和D个矩阵将被优化。对于这个计算，你应该给出经过良好校准的矩阵，以便(可能)得到更好的结果。
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH      # 在K矩阵中固定焦距。
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO     # 固定长宽比。
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST     # 去掉畸变。

        # 内参、畸变系数、平移向量、旋转矩阵
        retS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2,
                                                               ChessImaR.shape[::-1], self.criteria_stereo, flags)

        # 左内参矩阵、左畸变向量、右内参矩阵、右畸变向量、旋转矩阵、平移矩阵
        return K1, D1, K2, D2, R, T

    # ==================================================================== #

    # =========================== 双目校正 =============================== #
    # 获取畸变校正、立体校正、重投影矩阵
    def getRectifyTransform(self, width, height, K1, D1, K2, D2, R, T):
        # 得出进行立体矫正所需要的映射矩阵
        # 左校正变换矩阵、右校正变换矩阵、左投影矩阵、右投影矩阵、深度差异映射矩阵
        R_l, R_r, P_l, P_r, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2,
                                                                       (width, height), R, T,
                                                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        # # 标志CALIB_ZERO_DISPARITY，它用于匹配图像之间的y轴

        # 计算畸变矫正和立体校正的映射变换。
        map_lx, map_ly = cv2.initUndistortRectifyMap(K1, D1, R_l, P_l, (width, height), cv2.CV_32FC1)
        map_rx, map_ry = cv2.initUndistortRectifyMap(K2, D2, R_r, P_r, (width, height), cv2.CV_32FC1)

        return map_lx, map_ly, map_rx, map_ry, Q

    # 得到畸变校正和立体校正后的图像
    def get_rectify_img(self, imgL, imgR, map_lx, map_ly, map_rx, map_ry):
        rec_img_L = cv2.remap(imgL, map_lx, map_ly, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)  # 使用remap函数完成映射
        rec_img_R = cv2.remap(imgR, map_rx, map_ry, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        return rec_img_L, rec_img_R

    # 立体校正检验——极线对齐
    def draw_line(self, rec_img_L, rec_img_R):
        # 建立输出图像
        width = max(rec_img_L.shape[1], rec_img_R.shape[1])
        height = max(rec_img_L.shape[0], rec_img_R.shape[0])

        output = np.zeros((height, width * 2, 3), dtype=np.uint8)
        output[0:rec_img_L.shape[0], 0:rec_img_L.shape[1]] = rec_img_L
        output[0:rec_img_R.shape[0], rec_img_L.shape[1]:] = rec_img_R

        # 绘制等间距平行线
        line_interval = 50  # 直线间隔：50
        for k in range(height // line_interval):
            cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)

        return output  # 可显示的图像
    # ===================================================================== #


# 命令行参数解析
def get_parser():
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--width', type=int, default=11, help='chessboard width size')
    parser.add_argument('--height', type=int, default=8, help='chessboard height size')
    parser.add_argument('--lattice', type=float, default=0.035, help='lattice length')
    parser.add_argument('--winSize', type=int, default=3, help='pixel window size')
    parser.add_argument('--image_dir', type=str, default="./data/", help='images path')
    parser.add_argument('--save_dir', type=str, default="./result/", help='path to save file')
    parser.add_argument('--file_name', type=str, default="camera_params", help='camera params save file')
    return parser


# 获取双目相机图片路径
def get_file(path):  # 获取文件路径
    img_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            img_path.append(os.path.join(root, file))

    # 根据文件名排序（将文件名转换为整数进行排序）
    img_path = sorted(img_path, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    return img_path


if __name__ == "__main__":
    ### 读取双目相机的参数
    args = get_parser().parse_args()

    ### 读取当前路径下的所有图片路径
    left_files = get_file(args.image_dir + 'left')
    right_files = get_file(args.image_dir + 'right')
    print("左图路径:")
    for f in left_files:
        print(f)

    print("\n右图路径:")
    for f in right_files:
        print(f)

    ### 双目相机的标定
    calibration = Stereo_Camera_Calibration(args.width, args.height, args.lattice)

    ### 计算双目相机的 内参 外参
    left_K, left_D, right_K, right_D, R, T = calibration.stereo_calibration(left_files, right_files)

    # 计算立体校正所需的映射矩阵（map_lx, map_ly, map_rx, map_ry）和重投影矩阵 Q
    image_height, image_width = cv2.imread(left_files[0]).shape[0:2]
    map_lx, map_ly, map_rx, map_ry, Q = calibration.getRectifyTransform(image_width, image_height, left_K, left_D, right_K, right_D, R, T)

    # 查看校正效果
    # 读取target图像
    imgL = cv2.imread(left_files[2])
    imgR = cv2.imread(right_files[2])

    img_ = calibration.draw_line(imgL, imgR)
    cv2.imshow("img", img_)
    rec_img_L, rec_img_R = calibration.get_rectify_img(imgL, imgR, map_lx, map_ly, map_rx, map_ry)
    img_show = calibration.draw_line(rec_img_L, rec_img_R)
    cv2.imshow("output", img_show)
    cv2.waitKey(0)

    # 保存成YAML文件
    # 保存
    params_dict = {}
    params_dict['size'] = [width, height]
    params_dict['K1'] = left_K.tolist()
    params_dict['D1'] = left_D.tolist()
    params_dict['K2'] = right_K.tolist()
    params_dict['D2'] = right_D.tolist()
    params_dict['map_lx'] = map_lx.tolist()
    params_dict['map_ly'] = map_ly.tolist()
    params_dict['map_rx'] = map_rx.tolist()
    params_dict['map_ry'] = map_ry.tolist()
    params_dict['R'] = R.tolist()
    params_dict['T'] = T.tolist()
    params_dict['Q'] = Q.tolist()

    # 保存相机参数为YAML文件
    file_path_yaml = args.save_dir + args.file_name + ".yaml"
    with open(file_path_yaml, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False)

    print("Camera parameters saved to YAML format!")
    print("ALL Make Done!")
