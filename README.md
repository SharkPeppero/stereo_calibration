# 双目相机内外参标定

## 双目相机内参标定
python3 script/stereo_capture.py --device zed

## 双目相机外参标定 图像对齐以及深度图计算
python3 script/stereo_calibration.py --img_dir ./data/20250510_180342/