"""调试可视化小工具：把深度图转为伪彩色、拼接图像等。"""
from __future__ import annotations

import cv2
import numpy as np


def depth_to_colormap(depth: np.ndarray, near: float | None = None,
                      far: float | None = None) -> np.ndarray:
    """深度图 -> 伪彩色 RGB，便于展示。"""
    d = depth.copy()
    finite = np.isfinite(d)
    if near is None:
        near = float(np.percentile(d[finite], 2))
    if far is None:
        far = float(np.percentile(d[finite], 98))
    norm = np.clip((d - near) / max(far - near, 1e-6), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def hstack_images(*imgs: np.ndarray, pad: int = 8, bg: int = 255) -> np.ndarray:
    """等高水平拼接若干图像。"""
    h = max(im.shape[0] for im in imgs)
    out = []
    for im in imgs:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h))
        out.append(im)
        out.append(np.full((h, pad, 3), bg, np.uint8))
    return np.hstack(out[:-1])
