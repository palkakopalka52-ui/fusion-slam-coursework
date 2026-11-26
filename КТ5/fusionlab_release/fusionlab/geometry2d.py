"""SE(2) helpers: p=[x (m), y (m), theta (rad)]."""
import numpy as np
from .slam_core import wrap, relative_pose, integrate_relative, range_bearing


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def pose_matrix(p):
    T = np.eye(3)
    T[:2, :2] = rotation(p[2])
    T[:2, 2] = p[:2]
    return T


def inverse(T):
    R, t = T[:2, :2], T[:2, 2]
    ans = np.eye(3)
    ans[:2, :2] = R.T
    ans[:2, 2] = -R.T @ t
    return ans


def fk2r(q, lengths=(1., 1.)):
    """Position (m) of a planar two-link arm tip, joint angles in radians."""
    T1 = pose_matrix([0., 0., q[0]]) @ pose_matrix([lengths[0], 0., 0.])
    T2 = pose_matrix([0., 0., q[1]]) @ pose_matrix([lengths[1], 0., 0.])
    return (T1 @ T2)[:2, 2]
