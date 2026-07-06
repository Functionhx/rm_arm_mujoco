"""颜色分割目标检测：在 HSV 空间分割红色能量单元，输出像素质心、掩膜与标注图。"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    found: bool
    u: float = 0.0            # 质心像素坐标
    v: float = 0.0
    area: float = 0.0
    bbox: tuple = (0, 0, 0, 0)  # x, y, w, h
    mask: np.ndarray | None = None


class ColorDetector:
    def __init__(self, vision_cfg: dict):
        self.lo1 = np.array(vision_cfg["hsv_lower1"])
        self.hi1 = np.array(vision_cfg["hsv_upper1"])
        self.lo2 = np.array(vision_cfg["hsv_lower2"])
        self.hi2 = np.array(vision_cfg["hsv_upper2"])
        self.min_area = vision_cfg["min_area"]

    def detect(self, rgb: np.ndarray) -> Detection:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo1, self.hi1) | cv2.inRange(hsv, self.lo2, self.hi2)
        # 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(found=False, mask=mask)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return Detection(found=False, mask=mask, area=area)
        M = cv2.moments(c)
        u = M["m10"] / (M["m00"] + 1e-9)
        v = M["m01"] / (M["m00"] + 1e-9)
        x, y, w, h = cv2.boundingRect(c)
        return Detection(found=True, u=u, v=v, area=area, bbox=(x, y, w, h), mask=mask)

    @staticmethod
    def annotate(rgb: np.ndarray, det: Detection) -> np.ndarray:
        """在 RGB 图上绘制检测框与质心，用于可视化。"""
        img = rgb.copy()
        if det.found:
            x, y, w, h = det.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, (int(det.u), int(det.v)), 5, (255, 255, 0), -1)
            cv2.putText(img, "energy_unit", (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img
