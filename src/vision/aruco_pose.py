"""ArUco / solvePnP 位姿估计（对照实现，可选）。

当目标上贴有 ArUco 标记时，可用本模块估计其 6D 位姿；本仿真主线采用
颜色分割 + RGB-D 反投影（见 pose_estimator.py），此处提供基于 PnP 的对照方案。
"""
from __future__ import annotations

import cv2
import numpy as np


class ArucoPoseEstimator:
    def __init__(self, K: np.ndarray, marker_length: float = 0.03,
                 dist_coeffs: np.ndarray | None = None):
        self.K = K
        self.marker_length = marker_length
        self.dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        # OpenCV 4.7+ ArUco API
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

    def estimate(self, rgb: np.ndarray):
        """返回 [(id, rvec, tvec), ...]，相机系下的标记位姿。"""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is None:
            return results
        half = self.marker_length / 2.0
        obj_pts = np.array([[-half, half, 0], [half, half, 0],
                            [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        for i, mid in enumerate(ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[i][0], self.K, self.dist)
            if ok:
                results.append((int(mid), rvec, tvec))
        return results
