import cv2
import numpy as np
import os
import glob
from hikvisionapi import Client

# 摄像头参数
CAM_IP = "10.24.4.118"
CAM_USER = "admin"
CAM_PWD = "smbusmbu333"
CHECKERBOARD_SIZE = (5, 7)
SQUARE_SIZE = 0.03

# 从摄像头获取图像
def cam_capture_frame(file_name='capture.jpg'):
    cam = Client(f"http://{CAM_IP}", CAM_USER, CAM_PWD, timeout=10)
    response = cam.Streaming.channels[102].picture(method='get', type='opaque_data')
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# 增强图像对比度和锐化
def preprocess_image(img, contrast=1.5, brightness=20):
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# 分步检测两对棋盘格
def detect_two_chessboards(img, chessboard_size=CHECKERBOARD_SIZE, attempts=3):
    height, width = img.shape[:2]
    left_half = img[:, :width // 2]
    right_half = img[:, width // 2:]

    for _ in range(attempts):
        processed_left = preprocess_image(left_half)
        processed_right = preprocess_image(right_half)

        ret_left, corners_left = cv2.findChessboardCorners(
            processed_left, chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            processed_right, chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret_left and ret_right:
            # 亚像素精度优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(processed_left, corners_left, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(processed_right, corners_right, (11, 11), (-1, -1), criteria)

            corners_right += np.array([width // 2, 0], dtype=np.float32)
            return corners_left, corners_right

    return None, None

# 计算摄像头校准矩阵
def cal_calibration_matrix(chessboard_size=CHECKERBOARD_SIZE, square_size=SQUARE_SIZE):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints, imgpoints = [], []
    images = glob.glob('calibration_images/*.jpg')
    if not images:
        raise FileNotFoundError("No calibration images in folder './calibration_images/'.")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners_left, corners_right = detect_two_chessboards(gray)

        if corners_left is not None and corners_right is not None:
            objpoints.extend([objp, objp])
            imgpoints.extend([corners_left, corners_right])

            # 显示角点
            cv2.drawChessboardCorners(img, chessboard_size, corners_left, True)
            cv2.drawChessboardCorners(img, chessboard_size, corners_right, True)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)
        else:
            print(f"Skipping image {fname}: unable to detect both chessboards.")

    cv2.destroyAllWindows()
    # 使用复杂畸变模型（8个畸变系数）
    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None,
        flags=cv2.CALIB_RATIONAL_MODEL
    )

    print("校准矩阵 (camera matrix):\n", mtx)
    print("畸变系数 (distortion coefficients):\n", dist)
    os.makedirs('matrix', exist_ok=True)
    np.save('matrix/calibration_mtx.npy', mtx)
    np.save('matrix/calibration_dist.npy', dist)

    return mtx, dist

# 使用校准矩阵进行图像校正，并裁剪边缘
def cam_calibration(mtx=None, dist=None, plot=False):
    if mtx is None or dist is None:
        mtx = np.load('matrix/calibration_mtx.npy')
        dist = np.load('matrix/calibration_dist.npy')

    img = cv2.imread('capture.jpg')
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 畸变校正
    dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

    # 裁剪校正后的图像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('capture_fixed.jpg', dst)

    if plot:
        cv2.imshow('Calibrated Image', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    for _ in range(10):
        cam_capture_frame(f'calibration_images/cal_{_}.jpg')

    mtx, dist = cal_calibration_matrix()
    cam_capture_frame('capture.jpg')
    cam_calibration(mtx, dist, plot=True)
