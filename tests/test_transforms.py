"""几何变换单元测试。"""
import numpy as np

from src.geometry.transforms import (
    compose, invert_transform, make_transform, matrix_to_quat_wxyz,
    quat_wxyz_to_matrix, pose_error, rpy_to_matrix, transform_point,
)


def test_quat_matrix_roundtrip():
    R = rpy_to_matrix(0.3, -0.5, 1.2)
    q = matrix_to_quat_wxyz(R)
    R2 = quat_wxyz_to_matrix(q)
    assert np.allclose(R, R2, atol=1e-9)


def test_invert_transform():
    R = rpy_to_matrix(0.2, 0.4, -0.7)
    T = make_transform(R, np.array([0.1, -0.2, 0.3]))
    Ti = invert_transform(T)
    assert np.allclose(T @ Ti, np.eye(4), atol=1e-9)


def test_transform_point_roundtrip():
    R = rpy_to_matrix(1.0, 0.2, -0.3)
    T = make_transform(R, np.array([0.5, 0.5, 0.5]))
    p = np.array([0.1, 0.2, 0.3])
    p2 = transform_point(invert_transform(T), transform_point(T, p))
    assert np.allclose(p, p2, atol=1e-9)


def test_compose_identity():
    R = rpy_to_matrix(0.1, 0.2, 0.3)
    T = make_transform(R, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(compose(T, invert_transform(T)), np.eye(4), atol=1e-9)


def test_pose_error_zero():
    T = make_transform(rpy_to_matrix(0.1, 0.2, 0.3), np.array([0.1, 0.2, 0.3]))
    e = pose_error(T, T)
    assert np.allclose(e, np.zeros(6), atol=1e-9)
