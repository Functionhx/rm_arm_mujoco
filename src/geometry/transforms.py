"""SE(3) / 旋转变换工具。

统一约定：
- 齐次变换矩阵 T 为 4x4 numpy 数组。
- 四元数采用 MuJoCo 约定 [w, x, y, z]。
- 旋转矩阵 R 为 3x3。
内部使用 scipy 的 Rotation 保证数值稳健。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------- 四元数 <-> 旋转矩阵
def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """MuJoCo 四元数 [w,x,y,z] -> 3x3 旋转矩阵。"""
    w, x, y, z = quat
    r = Rotation.from_quat([x, y, z, w])  # scipy 用 [x,y,z,w]
    return r.as_matrix()


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> MuJoCo 四元数 [w,x,y,z]。"""
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """固定轴 XYZ 欧拉角 (rad) -> 旋转矩阵。"""
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_euler("xyz")


# ---------------------------------------------------------------- 齐次变换
def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """由旋转 R (3x3) 与平移 p (3,) 组装 4x4 齐次变换。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p).reshape(3)
    return T


def transform_from_quat_pos(quat: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return make_transform(quat_wxyz_to_matrix(quat), pos)


def invert_transform(T: np.ndarray) -> np.ndarray:
    """4x4 齐次变换求逆（解析）。"""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def compose(*transforms: np.ndarray) -> np.ndarray:
    """按顺序连乘多个齐次变换：compose(A, B, C) = A @ B @ C。"""
    out = np.eye(4)
    for T in transforms:
        out = out @ T
    return out


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """用齐次变换 T 变换一个 3D 点。"""
    p = np.ones(4)
    p[:3] = np.asarray(point).reshape(3)
    return (T @ p)[:3]


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """批量变换 (N,3) 点集。"""
    points = np.asarray(points)
    ph = np.hstack([points, np.ones((points.shape[0], 1))])
    return (ph @ T.T)[:, :3]


# ---------------------------------------------------------------- 位姿误差
def rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """姿态误差的轴角向量（3,），可用作 IK 的旋转误差项。"""
    R_err = R_target @ R_current.T
    return Rotation.from_matrix(R_err).as_rotvec()


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """6D 位姿误差 [位置(3), 旋转轴角(3)]。"""
    e_pos = T_target[:3, 3] - T_current[:3, 3]
    e_rot = rotation_error(T_current[:3, :3], T_target[:3, :3])
    return np.hstack([e_pos, e_rot])
