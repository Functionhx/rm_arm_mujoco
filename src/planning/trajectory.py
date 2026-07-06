"""轨迹规划：关节空间五次多项式插值 + 笛卡尔直线路径点。"""
from __future__ import annotations

import numpy as np


def quintic_scaling(num: int) -> np.ndarray:
    """生成 [0,1] 上的五次多项式时间标度 s(t)，两端速度、加速度为 0。"""
    t = np.linspace(0.0, 1.0, num)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def joint_trajectory(q_start: np.ndarray, q_goal: np.ndarray, num: int = 60) -> np.ndarray:
    """关节空间五次多项式轨迹，返回 (num, ndof)。"""
    s = quintic_scaling(num)
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    return q_start[None, :] + s[:, None] * (q_goal - q_start)[None, :]


def cartesian_line(p_start: np.ndarray, p_goal: np.ndarray, num: int = 40) -> np.ndarray:
    """笛卡尔直线路径点（位置），五次时间标度，返回 (num, 3)。"""
    s = quintic_scaling(num)
    p_start = np.asarray(p_start)
    p_goal = np.asarray(p_goal)
    return p_start[None, :] + s[:, None] * (p_goal - p_start)[None, :]
