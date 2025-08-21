#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time ： 2025/8/5 15:47
@Auth ： miaomiaosun
@File ：soft_rasterizer_main.py
@IDE ：PyCharm

"""
# !/usr/bin/env python
# coding: utf-8
from datetime import datetime
import json
import os
import time
from glob import glob

import cv2
import imageio
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, tree_util, lax
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import optax
import imageio.v3 as iio
import argparse
from pathlib import Path
from PIL import Image
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from sympy import Triangle
# 需加载预训练 VGG（JAX 版本，或用 flax 实现）
from jax.example_libraries import stax
from jax.example_libraries.stax import Conv, Relu, Flatten

# 启用64位精度以获得更稳定的梯度
jax.config.update("jax_enable_x64", True)
import random

# 设置随机种子
RANDOM_SEED = 42  # 可以任意选择，但固定
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# 设置JAX的随机种子
key = jax.random.PRNGKey(RANDOM_SEED)
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'


# 使用dataclass定义图元结构
@dataclass
class Circle:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    radius: jnp.ndarray  # 标量 (归一化单位)
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


@dataclass
class Rectangle:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    size: jnp.ndarray  # [width, height] (归一化单位)
    rotate_theta: jnp.ndarray  # 弧
    round: jnp.ndarray  # 圆角半径（归一化单位）
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


@dataclass
class Triangle:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    radius: jnp.ndarray  # [width, height] (归一化单位)
    rotate_theta: jnp.ndarray
    round: jnp.ndarray  # 圆角半径（归一化单位）
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


@dataclass
class Arc:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    shape_theta: jnp.ndarray  # 圆弧的角度范围（弧度）
    rotate_theta: jnp.ndarray
    radius: jnp.ndarray  # 标量 (归一化单位)
    round: jnp.ndarray  # 圆弧圆角半径
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


# @dataclass
# class Capsule:
#     center: jnp.ndarray  # [x, y] (归一化坐标系) 胶囊中心点坐标
#     radius: jnp.ndarray  # 胶囊线段部分的长度
#     rotate_theta: jnp.ndarray  # 旋转角度
#     round: jnp.ndarray  # 胶囊的宽度
#     color: jnp.ndarray
#     depth: jnp.ndarray  # 深度值，可优化
@dataclass
class Capsule:
    center: jnp.ndarray  # [x, y] (归一化坐标系) 胶囊中心点坐标
    length: jnp.ndarray  # 胶囊线段部分的长度
    a: jnp.ndarray  # 胶囊弯曲程度
    rotate_theta: jnp.ndarray  # 旋转角度
    round: jnp.ndarray  # 胶囊的宽度
    color: jnp.ndarray
    depth: jnp.ndarray  # 深度值，可优化


@dataclass
class Trapezoid:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    width1: jnp.ndarray  # 底宽
    width2: jnp.ndarray  # 顶宽
    height: jnp.ndarray  # 高
    rotate_theta: jnp.ndarray
    round: jnp.ndarray  # 圆角半径（归一化单位）
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


@dataclass
class Star:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    radius: jnp.ndarray  # 从五角星中心到其顶点（最尖端）的距离
    theta: jnp.ndarray  # 控制星形边角的尖锐程度，控制内半径比例（星尖到凹谷的距离比例）
    external_angle: jnp.ndarray  # 边角圆角半径,控制五角星角圆润程度
    round: jnp.ndarray  # 调整星形轮廓的形状参数，控制星形 “凸起程度” 与 “凹陷深度”
    k: jnp.ndarray  # 旋转角度（弧度）
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


@dataclass
class halfCircle:
    center: jnp.ndarray  # [x, y] (归一化坐标系)
    radius: jnp.ndarray  # 标量 (归一化单位)
    rotate_theta: jnp.ndarray
    round: jnp.ndarray
    color: jnp.ndarray  # [r, g, b, a]
    depth: jnp.ndarray  # 深度值，可优化


# 注册自定义类型为PyTree节点
tree_util.register_pytree_node(
    Circle,
    lambda c: ((c.center, c.radius, c.color, c.depth), None),
    lambda _, data: Circle(*data)
)

tree_util.register_pytree_node(
    Rectangle,
    lambda r: ((r.center, r.size, r.rotate_theta, r.round, r.color, r.depth), None),
    lambda _, data: Rectangle(*data)
)
tree_util.register_pytree_node(
    Triangle,
    lambda r: ((r.center, r.radius, r.rotate_theta, r.round, r.color, r.depth), None),
    lambda _, data: Triangle(*data)
)

tree_util.register_pytree_node(
    Arc,
    lambda c: ((c.center, c.shape_theta, c.rotate_theta, c.radius, c.round, c.color, c.depth), None),
    lambda _, data: Arc(*data)
)
# tree_util.register_pytree_node(
#     Capsule,
#     lambda c: ((c.center,c.radius,c.rotate_theta, c.round,c.color,c.depth), None),
#     lambda _, data: Capsule(*data)
# )
tree_util.register_pytree_node(
    Capsule,
    lambda c: ((c.center, c.length, c.a, c.rotate_theta, c.round, c.color, c.depth), None),
    lambda _, data: Capsule(*data)
)

tree_util.register_pytree_node(
    Trapezoid,
    lambda c: ((c.center, c.bottom_width, c.top_width, c.height, c.round_radius, c.rotate_theta, c.color, c.depth),
               None),
    lambda _, data: Trapezoid(*data)
)
tree_util.register_pytree_node(
    Star,
    lambda s: ((s.center, s.radius, s.theta, s.external_angle, s.round, s.k, s.color, s.depth), None),
    lambda _, data: Star(*data)
)

tree_util.register_pytree_node(
    halfCircle,
    lambda c: ((c.center, c.radius, c.rotate_theta, c.round, c.color, c.depth), None),
    lambda _, data: halfCircle(*data)
)


# ==============================
# 符号距离函数 (SDF) - 在归一化坐标系中
# ==============================
def Circle_sdf(center, radius, points):
    """圆形的符号距离函数，在归一化坐标系中"""
    diff = points - center
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    return dist - radius


def halfCircle_sdf(center, radius, rotate_theta, cut_offset, points):
    """
    计算旋转圆盘的SDF

    参数:
    center: 圆盘中心点 [x, y]
    radius: 圆盘半径
    rotate_theta: 圆盘的旋转角度（弧度）
    points: 需要计算SDF的点集，形状为(..., 2)

    返回:
    每个点到圆盘的符号距离
    """
    # 确保半径为有效值
    is_valid = radius > 1e-6

    def compute_valid_sdf():
        # 平移点到圆心坐标系
        translated = points - center

        # 计算旋转矩阵
        cos_theta = jnp.cos(rotate_theta)
        sin_theta = jnp.sin(rotate_theta)

        # 应用反向旋转（将点转换到圆盘的局部坐标系）
        x_rot = translated[..., 0] * cos_theta + translated[..., 1] * sin_theta
        y_rot = -translated[..., 0] * sin_theta + translated[..., 1] * cos_theta

        # 计算点到圆心的距离
        dist_to_center = jnp.sqrt(x_rot ** 2 + y_rot ** 2 + 1e-12)

        # 计算圆盘本身的SDF
        disk_sdf = dist_to_center - radius

        # 计算点到切割直线的距离
        line_sdf = x_rot - cut_offset

        # 组合两个SDF：取最大值表示交集
        # 当两个SDF都为负时，取最大值（离边界最近）
        # 当至少一个为正时，取最大值（到形状的距离）
        combined_sdf = jnp.maximum(disk_sdf, -line_sdf)

        # 处理切割直线超出圆盘范围的情况
        abs_cut_offset = jnp.abs(cut_offset)
        return jnp.where(
            abs_cut_offset > radius,
            # 如果切割直线完全在圆外，返回整个圆盘或空集
            jnp.where(cut_offset < 0, disk_sdf, jnp.full_like(disk_sdf, 1e10)),
            combined_sdf
        )

    def invalid_sdf():
        return jnp.full(points.shape[:-1], 1e10)

    # 根据半径有效性选择计算路径
    return jax.lax.cond(is_valid, compute_valid_sdf, invalid_sdf)


def rectangle_sdf(center, size, rotate_theta, round, points):
    """
    计算旋转圆角矩形的有符号距离场(SDF)

    参数:
        center: 矩形中心坐标 [cx, cy]
        size: 矩形尺寸 [width, height]
        rotate_theta: 旋转角度(弧度)
        round: 圆角半径
        points: 要计算的点集, shape=(N,2)

    返回:
        各点的SDF值, shape=(N,)
    """
    size = jnp.maximum(size, 0.0)  # 确保尺寸为正

    def compute_valid_sdf():
        x, y = center
        w, h = size
        # 获取旋转矩阵
        cos_theta = jnp.cos(rotate_theta)
        sin_theta = jnp.sin(rotate_theta)

        rotate_mat = jnp.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
        # 3. 平移向量 (2,)
        trans = jnp.array([x, y])

        # 4. 坐标变换：先平移到原点，再旋转（关键修复：确保维度对齐）
        # 对每个点应用变换：(x' = (x - cx)*cosθ + (y - cy)*sinθ, ...)
        translated = points - trans  # 平移到原点 (N, 2)
        new_coords = translated @ rotate_mat  # 应用旋转 (N, 2) @ (2, 2) = (N, 2)

        # 5. 计算到矩形边界的距离（含圆角）
        p0 = jnp.abs(new_coords[:, 0]) - w / 2 + round  # x方向距离 (N,)
        p1 = jnp.abs(new_coords[:, 1]) - h / 2 + round  # y方向距离 (N,)

        # 6. 计算外部距离（超出矩形的圆角距离）
        q0 = jnp.clip(p0, a_min=0.0)  # (N,)
        q1 = jnp.clip(p1, a_min=0.0)  # (N,)
        outside = jnp.sqrt(q0 ** 2 + q1 ** 2 + 1e-12)  # (N,)

        # 7. 计算内部距离（矩形内部的距离）
        inside = jnp.clip(jnp.maximum(p0, p1), a_max=0.0)  # (N,)

        # 8. 组合内外距离并应用圆角
        sdf = outside + inside - round
        return sdf

    def invalid_sdf():
        # 无效 size 的处理：返回极大 SDF
        return jnp.full(points.shape[:-1], 1e10)

    # 检查 size 是否有效（至少有一个维度足够大）
    is_valid = jnp.min(size) > 1e-6  # 使用稍大的阈值确保安全

    # 使用条件选择函数
    return jax.lax.cond(is_valid, compute_valid_sdf, invalid_sdf)


def triangle_sdf(center, radius, rotate_theta, round, points):
    x, y = center
    # 计算等边三角形边长（基于外接圆半径）

    pi = jnp.pi
    length = 2 * radius * jnp.cos(pi / 6)  # cos(30°) = √3/2
    cos_theta = jnp.cos(rotate_theta)
    sin_theta = jnp.sin(rotate_theta)
    # 旋转矩阵和坐标变换
    rotate_mat = jnp.array([[cos_theta, -sin_theta],
                            [sin_theta, cos_theta]])
    trans = jnp.array([x, y])

    translated = points - trans

    new_coords = translated @ rotate_mat  # 应用旋转 (N, 2) @ (2, 2) = (N, 2)

    # 应用旋转和平移（转换到三角形局部坐标系）
    # 提取x和y坐标
    coords0 = jnp.abs(new_coords[..., 0])  # x坐标取绝对值（利用对称性）
    coords1 = new_coords[..., 1]  # y坐标

    # 等边三角形的几何常数（√3）
    k = jnp.sqrt(3.0)
    r_half = length / 2

    # 初始化距离计算参数
    pp0 = coords0
    pp1 = coords1

    # 条件判断：确定需要特殊处理的区域（替代原PyTorch的index_put）
    cond = coords0 + k * coords1 > 0
    # 使用where实现向量化条件更新（避免循环和索引操作）
    updated0 = (coords0 - k * coords1) / 2
    updated1 = (-k * coords0 - coords1) / 2
    pp0 = jnp.where(cond, updated0, pp0)
    pp1 = jnp.where(cond, updated1, pp1)

    # 计算到三角形边界的距离
    pp0 = pp0 - r_half
    pp1 = pp1 + r_half / k

    # 处理x方向的约束
    pp0_clamp = jnp.clip(pp0, a_min=-length, a_max=0.0)
    pp0 = pp0 - pp0_clamp

    # 计算最终带符号的距离
    sdf = -jnp.linalg.norm(jnp.stack([pp0, pp1], axis=0), axis=0) * jnp.sign(pp1)

    # 应用圆角处理
    sdf = sdf - round
    return sdf


def capsule_sdf(center, radius, rotate_theta, round, points):
    """
    计算点集到胶囊形状的有向距离函数

    参数:
    points: 待计算距离的点集，形状为(n, d)，d为维度(2或3)
    center: 胶囊中心点坐标，形状为(d,)
    length: 胶囊线段部分的长度
    radius: 胶囊的半径

    返回:
    点集到胶囊的有向距离，形状为(n,)
    """
    x, y = center

    radius = jnp.maximum(radius, 1e-8)  # 避免半径为0导致数值问题
    length = 2 * round  # 胶囊体线段长度

    # 统一坐标格式为 (2, N)
    if points.shape[0] != 2:
        coords = points.T  # 转换为 (2, N) 方便矩阵运算

    # 构建旋转矩阵（顺时针为正方向时取负角度）
    cos_theta = jnp.cos(rotate_theta)
    sin_theta = jnp.sin(rotate_theta)
    rotate_mat = jnp.array([[cos_theta, -sin_theta],
                            [sin_theta, cos_theta]])  # 旋转矩阵

    # 定义局部坐标系下的两个端点（以中心为原点）
    endpoints_local = jnp.array([[0.0, -length / 2],
                                 [0.0, length / 2]]).T  # 形状 (2, 2)

    # 旋转端点并平移到世界坐标系
    endpoints = rotate_mat @ endpoints_local  # 应用旋转
    endpoints += jnp.array([[x], [y]])  # 应用平移（中心坐标）
    a = endpoints[:, 0:1]  # 起点坐标 (2, 1)
    b = endpoints[:, 1:2]  # 终点坐标 (2, 1)

    # 计算向量
    p = coords  # 点集 (2, N)
    pa = p - a  # 点到起点的向量 (2, N)
    ba = b - a  # 起点到终点的向量 (2, 1)

    # 计算投影比例h（限制在[0,1]范围内，确保在线段上）
    ba_dot_ba = jnp.sum(ba * ba) + 1e-12  # 线段长度平方（避免除零）
    ba_dot_pa = jnp.sum(ba * pa, axis=0, keepdims=True)  # 点积 (1, N)
    h = jnp.clip(ba_dot_pa / ba_dot_ba, 0.0, 1.0)  # 投影比例 (1, N)

    # 计算点到线段的最短距离，再减去胶囊体半径
    closest = a + ba * h  # 线段上最近点 (2, N)
    dist = jnp.linalg.norm(p - closest, axis=0)  # 点到线段的距离 (N,)
    sdf = dist - radius  # 胶囊体SDF
    return sdf


def arc_sdf(center, shape_theta, rotate_theta, radius, round_radius, points):
    """
    计算两端带相同弧度的圆弧SDF（适配新参数列表）

    新参数说明:
    center_x: 圆心X坐标
    center_y: 圆心Y坐标
    shape_theta: 圆弧张角（弧度）
    rotate_theta: 圆弧旋转角度（弧度）
    radius: 圆弧半径
    round_radius: 起点和终点的圆角半径（统一值）
    points: 待计算的点集
    """
    x, y = center
    ra = jnp.maximum(radius, 1e-8)  # 避免半径为0导致数值问题
    rb = jnp.maximum(round_radius, 0.0)  # 确保宽度非负

    # 统一坐标格式为 (2, N)
    if points.shape[0] != 2:
        coords = points.T  # 转换为 (2, N) 方便矩阵运算

    # 构建旋转矩阵（取转置等价于逆矩阵，实现反向旋转）
    cos_rot = jnp.cos(rotate_theta)
    sin_rot = jnp.sin(rotate_theta)
    rotate_mat_t = jnp.array([[cos_rot, sin_rot],  # 旋转矩阵的转置（用于反向旋转）
                              [-sin_rot, cos_rot]])

    # 平移向量（中心坐标）
    translation = jnp.array([x, y])

    # 坐标变换：先旋转再平移（抵消圆弧的旋转和平移，转换到局部坐标系）
    new_coords = rotate_mat_t @ coords - rotate_mat_t @ translation.reshape(2, 1)

    # 提取局部坐标系下的坐标分量（x0取绝对值，对称处理）
    coords0 = jnp.abs(new_coords[0])  # x分量取绝对值
    coords1 = new_coords[1]  # y分量

    # 计算张角对应的正弦和余弦（用于判断点是否在圆弧范围内）
    sc = jnp.array([jnp.sin(shape_theta), jnp.cos(shape_theta)])

    # 判断点是否在圆弧的角度范围内（mask为True时使用圆角SDF）
    mask = (sc[1] * coords0 - sc[0] * coords1) > 1e-8  # 加微小值避免数值抖动

    # 圆弧主体SDF：到圆心的距离与半径的差，再减去宽度
    dist_center = jnp.sqrt(coords0 ** 2 + coords1 ** 2 + 1e-12)  # 到圆心的距离
    d1 = jnp.abs(dist_center - ra) - rb  # 主体SDF

    # 圆弧端点圆角SDF：以圆弧终点为圆心的圆
    arc_center = (ra * sc).reshape(2, 1)  # 圆角圆心坐标
    dx = coords0 - arc_center[0]  # x方向距离
    dy = coords1 - arc_center[1]  # y方向距离
    dist_round = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-12)  # 到圆角圆心的距离
    d2 = dist_round - rb  # 圆角SDF

    # 组合SDF：根据mask选择主体或圆角SDF
    sdf = jnp.where(mask, d2, d1)
    return sdf

def smoothstep(edge0, edge1, x):
    """JAX版本的平滑阶梯函数，与原PyTorch逻辑一致"""
    t = jnp.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def curved_capsule_sdf(center, length, a, rotate_theta, round_param, points):
    cx, cy = center
    # --- 角度修正 ---
    theta = (rotate_theta - jnp.pi / 2) % (jnp.pi * 2)

    # --- 坐标变换 ---
    if points.shape[0] != 2:
        points = points.T  # (2, N)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = jnp.array([[cos_theta, -sin_theta],
                   [sin_theta, cos_theta]])
    translation = jnp.array([cx, cy]).reshape(2, 1)
    p = R.T @ (points - translation)
    px, py = p[0, :], p[1, :]

    # ---- 线段SDF ----
    clamped_x = jnp.clip(px, -length, length)
    d_line = jnp.sqrt((px - clamped_x) ** 2 + py ** 2 + 1e-12)

    # ---- 曲率方向和大小 ----
    eps1 = 1e-6
    # abs_a = jnp.clip(jnp.abs(a), eps1)  # 曲率大小
    abs_a = jnp.sqrt(a * a + eps1)
    sign_a = jnp.where(a >= 0, 1.0, -1.0)  # 曲率方向

    # ---- 圆弧参数 ----
    scx = jnp.cos(abs_a / 2.0)
    scy = jnp.sin(abs_a / 2.0)
    ra = jnp.clip(length / abs_a, 0.0, 1e3)  # 半径

    # ---- 圆弧中心下移（方向依赖 sign_a）----
    py2 = py - sign_a * ra
    px2 = jnp.abs(px)

    # ---- 反射计算 ----
    dot_sp = scx * px2 + sign_a * scy * py2
    m = jnp.clip(dot_sp, a_min=0.0)
    qx = px2 - 2.0 * scx * m
    qy = py2 - 2.0 * sign_a * scy * m

    # ---- 圆弧距离 ----
    qlen = jnp.sqrt(qx ** 2 + qy ** 2)
    u = jnp.abs(ra) - qlen
    d_arc_left = jnp.sqrt(qx ** 2 + (qy + sign_a * ra) ** 2 + 1e-12)
    d_arc = jnp.where(qx < 0.0, d_arc_left, jnp.abs(u))

    # ---- 平滑过渡权重 ----
    a_abs = jnp.abs(a)
    curvature_based_eps = 0.1 * (length / (round_param + 1e-8))
    curvature_based_eps = jnp.clip(curvature_based_eps, 1e-3, 1e-1)
    transition_center = curvature_based_eps
    transition_width = curvature_based_eps * 0.2
    transition_center = 1e-3  # 固定很小
    transition_width = 5e-4
    x = (a_abs - transition_center) / (transition_width + 1e-8)
    k = 0.5 * (jnp.tanh(x) + 1.0)
    k = jnp.clip(k, 0.0, 1.0)

    # ---- 融合线段和圆弧 ----
    d = (1.0 - k) * d_line + k * d_arc
    sdf = d - round_param
    return sdf


def trapezoid_sdf(center, bottom_width, top_width, height, rotate_theta, round_radius, points):
    """
    带圆角的等边梯形有向距离函数

    参数:
        center: 梯形中心点坐标 (x, y)
        width: 梯形下底宽度
        height: 梯形高度
        top_width_ratio: 上底宽度与下底宽度的比例 (0 < ratio < 1)
        corner_radius: 圆角半径
        points: 待计算距离的点集，形状为(n, 2)

    返回:
        点集到梯形的有向距离，形状为(n,)
    """
    # 1. 将点转换到梯形局部坐标系
    bottom_width = jnp.maximum(bottom_width, 1e-3)  # 最小下底宽度
    top_width = jnp.maximum(top_width, 1e-3)  # 最小上底宽度
    height = jnp.maximum(height, 1e-3)  # 最小高度

    scale_factor = jnp.sqrt(bottom_width * height)  # 基于面积的尺度
    # 确保梯形形状有效性
    bottom_width = jnp.maximum(bottom_width, top_width + 0.05)  # 减小约束强度，提高灵活性
    # 3. 几何参数计算（使用尺度因子标准化）
    r1 = (bottom_width / 2.0) / scale_factor  # 下底半宽（标准化）
    r2 = (top_width / 2.0) / scale_factor  # 上底半宽（标准化）
    h = (height / 2.0) / scale_factor  # 半高（标准化）
    radius = round_radius / scale_factor  # 圆角半径（标准化）
    # 4. 坐标变换
    translated = points - jnp.array(center)
    cos_t = jnp.cos(rotate_theta)
    sin_t = jnp.sin(rotate_theta)
    rot_mat = jnp.array([[cos_t, -sin_t], [sin_t, cos_t]])
    local = translated @ rot_mat / scale_factor  # 坐标也进行标准化

    coords0 = jnp.abs(local[:, 0])
    coords1 = local[:, 1]

    # 5. 距离计算
    # 上/下边界距离
    upper_mask = coords1 <= 0.0
    qp1_x_upper = jnp.maximum(coords0 - r2, 0.0)
    qp1_x_lower = jnp.maximum(coords0 - r1, 0.0)
    qp1_x = jnp.where(upper_mask, qp1_x_upper, qp1_x_lower)
    qp1_y = jnp.abs(coords1) - h
    qp1 = jnp.stack([qp1_x, qp1_y], axis=-1)

    # 斜边距离
    ap = jnp.stack([coords0 - r1, coords1 - h], axis=-1)
    ab = jnp.array([r2 - r1, -2 * h])
    ab_dot_ab = jnp.dot(ab, ab) + 1e-8  # 避免除以零

    t = jnp.clip((ap @ ab) / ab_dot_ab, 0.0, 1.0)
    proj = ab * t[:, None]
    qp2 = ap - proj

    # 内部判断与距离计算
    inside = (qp1_y < 0) & (qp2[:, 0] < 0)
    sign = jnp.where(inside, -1.0, 1.0)
    d = sign * jnp.minimum(jnp.linalg.norm(qp1, axis=-1), jnp.linalg.norm(qp2, axis=-1))

    # 6. 圆角处理与尺度还原
    sdf = (d - radius) * scale_factor  # 还原尺度

    return sdf


def star_sdf(center, radius, theta, external_angle, round, k, points):
    cx, cy = center
    r = jnp.maximum(radius, 1e-8)
    round_param = jnp.maximum(round, 0.0)
    k = jnp.maximum(k, 1e-8)  # 避免除以零
    n = 5.0  # 五角星（固定为5个角，可根据需要修改）

    # 统一坐标格式为 (2, N)
    if points.shape[0] != 2:
        coords = points.T  # 转换为 (2, N) 方便矩阵运算

    # 步骤1：转换到局部坐标系并应用旋转
    # 平移到以中心为原点
    p = coords - jnp.array([[cx], [cy]])  # 形状 (2, N)

    # 构建旋转矩阵（反向旋转抵消星形自身旋转）
    cos_theta = jnp.cos(-theta)
    sin_theta = jnp.sin(-theta)
    rot = jnp.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])  # 旋转矩阵 (2, 2)
    p = rot @ p  # 应用旋转 (2, N)

    # 步骤2：预计算常量
    pi = jnp.pi
    m = n + external_angle * (2.0 - n)
    an = pi / n  # 角间距
    en = pi / m  # 边缘斜率角度

    # 星形顶点坐标和边缘斜率向量
    racs = r * jnp.array([jnp.cos(an), jnp.sin(an)])  # 顶点坐标 (2,)
    ecs = jnp.array([jnp.cos(en), jnp.sin(en)])  # 边缘斜率向量 (2,)

    # 步骤3：对称性处理 - 沿y轴反射（只计算右半部分，左半部分对称）
    p = jnp.stack([jnp.abs(p[0, :]), p[1, :]], axis=0)  # (2, N)

    # 步骤4：计算角度并确定所在扇形区域
    angle = jnp.arctan2(p[0, :], p[1, :])  # 计算角度（注意x,y顺序与atan2的参数对应）
    bn = (angle + 2.0 * pi) % (2.0 * an) - an  # 标准化到[-an, an]范围
    q = jnp.abs(jnp.sin(bn))  # 角度的正弦绝对值

    # 步骤5：转换到扇形局部坐标系
    length_p = jnp.linalg.norm(p, axis=0)  # 点到原点的距离 (N,)
    px = length_p * jnp.cos(bn)  # 局部x坐标 (N,)
    k = jnp.maximum(k, 1e-6)  # 确保分母不为零
    q_clamped = jnp.clip(q, 0.0, k)  # 限制q的范围
    # 局部y坐标（包含边缘平滑处理）
    py = length_p * (q + 0.5 * jnp.square(jnp.maximum(k - q_clamped, 0.0)) / k)  # (N,)

    p = jnp.stack([px, py], axis=0)  # (2, N)

    # 步骤6：线段SDF计算（星形边缘）
    p = p - racs.reshape(2, 1)  # 平移到顶点为原点 (2, N)
    dot = jnp.einsum("i,ij->j", ecs, p)  # 点与边缘斜率的点积 (N,)

    # 计算钳位范围
    max_val = racs[1] / ecs[1]  # 最大钳位值
    min_val = 0.0  # 最小钳位值
    clamped = jnp.clip(-dot, min_val, max_val)  # 钳位处理 (N,)

    # 计算最近点并更新距离
    p = p + ecs.reshape(2, 1) * clamped  # (2, N)

    # 最终SDF计算（减去圆角半径）
    sdf = jnp.linalg.norm(p, axis=0) * jnp.sign(p[0, :]) - round_param  # (N,)
    return sdf


def differentiable_rasterize(primitives, grid, softness=150, is_final_render=False):
    """
    可微光栅化函数
    输入:
        primitives: 图元列表
        grid: 像素网格坐标 (H, W, 2) - 归一化坐标系
        softness: 软化参数控制梯度传播
    返回:
        rgba图像 (H, W, 4)
    """

    H, W, _ = grid.shape
    points = grid.reshape(-1, 2)  # 展平像素坐标 [N, 2]

    # 提取所有图元的深度和颜色
    depths = jnp.array([p.depth for p in primitives])  # [M]
    colors = jnp.array([p.color for p in primitives])  # [M, 4]
    base_colors = colors[:, :3]  # RGB [M, 3]
    base_alphas = colors[:, 3]  # Alpha [M]

    # 计算每个图元在每个像素的SDF
    sdf_values = []
    for prim in primitives:
        # print('prim:',prim)
        if isinstance(prim, Circle):
            sdf = Circle_sdf(prim.center, prim.radius, points)
        elif isinstance(prim, Rectangle):
            sdf = rectangle_sdf(prim.center, prim.size, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Triangle):
            sdf = triangle_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Capsule):
            # sdf = capsule_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
            sdf = curved_capsule_sdf(prim.center, prim.length, prim.a, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Arc):
            sdf = arc_sdf(prim.center, prim.shape_theta, prim.rotate_theta, prim.radius, prim.round, points)
        elif isinstance(prim, Trapezoid):
            sdf = trapezoid_sdf(prim.center, prim.width1, prim.width2, prim.height, prim.rotate_theta, prim.round,
                                points)
        elif isinstance(prim, Star):
            sdf = star_sdf(prim.center, prim.radius, prim.theta, prim.external_angle, prim.round, prim.k, points)
        elif isinstance(prim, halfCircle):
            sdf = star_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
        sdf_values.append(sdf)

    sdf_matrix = jnp.stack(sdf_values, axis=0)  # [M, N]

    sdf_clamped = jnp.clip(sdf_matrix, -50.0, 50.0)
    softness = jnp.where(len(primitives) > 10, 100, 150)
    coverage_alpha = jax.nn.sigmoid(-sdf_clamped * softness)  # [M, N]
    # 计算最终alpha = 基础alpha × 覆盖度alpha
    final_alpha = base_alphas[:, jnp.newaxis] * coverage_alpha  # [M, N]

    # 计算深度权重 (逐像素softmax)
    depth_weights = jax.nn.softmax(depths * 0.01, axis=0)  # [M, N]
    # # 计算像素级权重 = 深度权重 × 最终alpha
    weights = depth_weights[:, jnp.newaxis] * final_alpha

    # # 归一化权重 (每个像素的所有图元权重和为1)
    weight_sum = jnp.sum(weights, axis=0) + 1e-4  # [N]
    normalized_weights = weights / weight_sum  # [M, N]

    # 计算预乘颜色 (RGB × alpha)
    premultiplied_rgb = base_colors[:, :, jnp.newaxis] * final_alpha[:, jnp.newaxis, :]  # [M, 3, N]

    # 混合颜色 (加权求和)
    # [M, 3, N] * [M, 1, N] -> 求和M -> [3, N]
    weighted_rgb = jnp.sum(premultiplied_rgb * normalized_weights[:, jnp.newaxis, :], axis=0)

    # 计算最终alpha通道 (加权求和)
    weighted_alpha = jnp.sum(final_alpha * normalized_weights, axis=0)  # [N]
    # 组合RGBA
    rgba = jnp.vstack([weighted_rgb, weighted_alpha]).T  # [N, 4]
    rgba_image = rgba.reshape(H, W, 4)
    return jnp.clip(rgba_image, 0, 1)


# ==============================
# 辅助函数 - 创建归一化网格
# ==============================
def create_normalized_grid(height, width):
    """创建归一化像素坐标系网格 (0,0)在左上角，(1,1)在右下角"""
    x = jnp.linspace(0, 1, width)
    y = jnp.linspace(0, 1, height)
    grid_x, grid_y = jnp.meshgrid(x, y)
    return jnp.stack([grid_x, grid_y], axis=-1)


# ==============================
# 参数加载和保存
# ==============================
def load_parameters(file_path):
    """从文本文件加载图元参数 - 归一化坐标系"""
    primitives = []
    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    # 提取所需字段
    shapes_id = data['shapes id']
    shapes_op = data['shapes op']
    shapes_params = data['shapes params']
    for i in range(len(shapes_id)):
        # 获取基本信息
        prim_type = shapes_id[i]
        sop = shapes_op[i]
        sparams = shapes_params[i]
        # 根据op确定颜色
        if sop == 1:  # 减集
            r, g, b = 0, 0, 0  # 黑色
        else:  # 并集(0)或交集(2)
            r, g, b = 1, 1, 1  # 白色

        # depth按顺序从1开始
        depth = i + 1
        depth = jnp.array(depth, dtype=jnp.float64)
        color = jnp.array([
            np.clip(r, 0.001, 0.999),
            np.clip(g, 0.001, 0.999),
            np.clip(b, 0.001, 0.999),
            0.999
        ])
        if prim_type == 0:
            # 格式: circle cx_norm cy_norm r_norm cr cg cb [ca]
            center_x, center_y, radius = map(float, sparams)
            center = jnp.array([center_x, center_y])
            primitives.append(Circle(center, radius, color, depth))

        elif prim_type == 1:
            center_x, center_y, width, height, rotate_theta, round = map(float, sparams)
            # 确保参数在有效范围内
            center_x = jnp.clip(center_x, 0.0, 1.0)
            center_y = jnp.clip(center_y, 0.0, 1.0)
            width = jnp.maximum(width, 0.0)
            height = jnp.maximum(height, 0.0)

            center = jnp.array([center_x, center_y])
            size = jnp.array([width, height])
            primitives.append(Rectangle(center, size, rotate_theta, round, color, depth))
        elif prim_type == 2:
            center_x, center_y, radius, rotate_theta, round = map(float, sparams)
            center = jnp.array([center_x, center_y])

            primitives.append(Triangle(center, radius, rotate_theta, round, color, depth))
        elif prim_type == 3:
            # center_x, center_y, round, rotate_theta,radius = map(float, sparams)
            # center = jnp.array([center_x, center_y])
            # primitives.append(Capsule(center, radius, rotate_theta, round,  color, depth))
            center_x, center_y, length, a, rotate_theta, round = map(float, sparams)
            center = jnp.array([center_x, center_y])
            primitives.append(Capsule(center, length, a, rotate_theta, round, color, depth))
        elif prim_type == 4:
            center_x, center_y, shape_theta, rotate_theta, radius, round = map(float, sparams)
            # 确保参数在有效范围内
            center_x = jnp.clip(center_x, 0.0, 1.0)
            center_y = jnp.clip(center_y, 0.0, 1.0)
            center = jnp.array([center_x, center_y])
            primitives.append(Arc(center, jnp.array(shape_theta), jnp.array(rotate_theta), radius, round,
                                  color, depth))
        elif prim_type == 5:
            center_x, center_y, width1, width2, height, rotate_theta, round = map(float, sparams)
            center = jnp.array([center_x, center_y])
            primitives.append(Trapezoid(center, width1, width2, height, rotate_theta, round, color, depth))
        elif prim_type == 6:
            center_x, center_y, radius, theta, external_angle, rotate_theta, round = map(float, sparams)

            # 确保参数在有效范围内
            center_x = jnp.clip(center_x, 0.0, 1.0)
            center_y = jnp.clip(center_y, 0.0, 1.0)
            center = jnp.array([center_x, center_y])
            # 创建五角星图元
            primitives.append(Star(center, radius, theta, external_angle, rotate_theta, round, color, depth))
        elif prim_type == 7:
            center_x, center_y, radius, rotate_theta, round = map(float, sparams)
            center = jnp.array([center_x, center_y])
            primitives.append(halfCircle(center, radius, rotate_theta, round, color, depth))

    return primitives


def save_parameters(file_path, primitives, height, width):
    """保存图元参数到文本文件 - 归一化坐标系"""
    with open(file_path, 'w') as f:
        for prim in primitives:
            if isinstance(prim, Circle):
                cx_norm, cy_norm = prim.center
                r_norm = prim.radius
                cr, cg, cb, ca = prim.color
                depth = prim.depth
                f.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {r_norm:.6f} "
                        f"{cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")

            elif isinstance(prim, Rectangle):
                cx_norm, cy_norm = prim.center
                w_norm, h_norm = prim.size
                rotate_theta = prim.rotate_theta
                round = prim.round
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(f"1 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f} "
                        f"{rotate_theta:.6f} {round:.6f} {cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")
            elif isinstance(prim, Triangle):
                cx_norm, cy_norm = prim.center
                radius = prim.radius
                rotate_theta = prim.rotate_theta
                round = prim.round
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(f"2 {cx_norm:.6f} {cy_norm:.6f} {radius:.6f} "
                        f"{rotate_theta:.6f} {round:.6f} {cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")

            elif isinstance(prim, Capsule):
                # cx_norm, cy_norm = prim.center
                # radius = prim.radius
                # round = prim.round
                # rotate_theta = prim.rotate_theta
                # cr, cg, cb, ca = prim.color
                # depth = prim.depth
                #
                # f.write(f"3 {cx_norm:.6f} {cy_norm:.6f} {radius:.6f} {rotate_theta:.6f} {round:.6f} "
                #         f"{cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")
                cx_norm, cy_norm = prim.center
                length = prim.length
                a = prim.a
                round = prim.round
                rotate_theta = prim.rotate_theta
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(f"3 {cx_norm:.6f} {cy_norm:.6f} {length:.6f} {a:.6f} {rotate_theta:.6f} {round:.6f} "
                        f"{cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")
            elif isinstance(prim, Arc):
                cx_norm, cy_norm = prim.center
                radius = prim.radius
                round = prim.round
                rotate_theta = prim.rotate_theta
                shape_theta = prim.shape_theta
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(f"4 {cx_norm:.6f} {cy_norm:.6f} {shape_theta:.6f} {rotate_theta:.6f} {radius:.6f} {round:.6f} "
                        f" {cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")
            elif isinstance(prim, Trapezoid):
                cx_norm, cy_norm = prim.center
                width1 = prim.width1
                width2 = prim.width2
                height = prim.height
                rotate_theta = prim.rotate_theta
                round = prim.round
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(
                    f"5 {cx_norm:.6f} {cy_norm:.6f} {width1:.6f} {width2:.6f} {height:.6f} {rotate_theta:.6f} {round:.6f}"
                    f"{cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")
            elif isinstance(prim, Star):
                cx_norm, cy_norm = prim.center
                radius = prim.radius
                theta = prim.theta
                external_angle = prim.external_angle
                round = prim.round
                k = prim.k
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(f"6 {cx_norm:.6f} {cy_norm:.6f} {radius:.6f} {theta:.6f} "
                        f"{external_angle:.6f} {round:.6f} {k:.6f} {cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")
            elif isinstance(prim, halfCircle):
                cx_norm, cy_norm = prim.center
                radius = prim.radius
                rotate_theta = prim.rotate_theta
                round = prim.round
                cr, cg, cb, ca = prim.color
                depth = prim.depth

                f.write(
                    f"7 {cx_norm:.6f} {cy_norm:.6f} {radius:.6f} {rotate_theta:.6f} {round:.6f} {cr:.6f} {cg:.6f} {cb:.6f} {ca:.6f} {depth:.6f}\n")


def edge_consistency_loss(rendered_img, target_img, mask=None):
    """
    边缘一致性损失

    参数:
        rendered_img: 渲染图像 (H, W, 3)
        target_img: 目标图像 (H, W, 3)
        mask: 可选掩码
    """
    if mask is None:
        mask = jnp.ones_like(rendered_img[..., 0])

    # 使用Sobel算子检测边缘
    def detect_edges(img):
        # 转换为灰度图
        gray = jnp.mean(img, axis=-1)

        # Sobel算子
        sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

        # 计算梯度
        grad_x = jax.scipy.signal.convolve2d(gray, sobel_x, mode='same')
        grad_y = jax.scipy.signal.convolve2d(gray, sobel_y, mode='same')

        # 边缘强度
        edges = jnp.sqrt(grad_x ** 2 + grad_y ** 2)
        return edges

    rendered_edges = detect_edges(rendered_img)
    target_edges = detect_edges(target_img)
    mask_edges = mask

    # 边缘一致性损失
    edge_loss = jnp.mean(jnp.abs(rendered_edges - target_edges) * mask_edges)

    return edge_loss


def intersection_aware_loss(primitives, grid, target_rgb, rendered_rgb):
    """
    交点感知损失函数 - 专门优化图元重叠区域

    参数:
        primitives: 图元列表
        grid: 像素网格 (H, W, 2)
        target_img: 目标图像 (H, W, 3)
        rendered_img: 渲染图像 (H, W, 4)

    返回:
        交点区域损失值
    """
    H, W, _ = grid.shape
    points = grid.reshape(-1, 2)

    # 1. 计算每个图元的SDF
    sdf_matrix = []
    for prim in primitives:
        if isinstance(prim, Circle):
            sdf = Circle_sdf(prim.center, prim.radius, points)
        elif isinstance(prim, Rectangle):
            sdf = rectangle_sdf(prim.center, prim.size, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Triangle):
            sdf = triangle_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Capsule):
            # sdf = capsule_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
            sdf = curved_capsule_sdf(prim.center, prim.length, prim.a, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Arc):
            sdf = arc_sdf(prim.center, prim.shape_theta, prim.rotate_theta, prim.radius, prim.round, points)
        elif isinstance(prim, Trapezoid):
            sdf = trapezoid_sdf(prim.center, prim.width1, prim.width2, prim.height, prim.rotate_theta, prim.round,
                                points)
        elif isinstance(prim, Star):
            sdf = star_sdf(prim.center, prim.radius, prim.theta, prim.external_angle, prim.round, prim.k, points)
        elif isinstance(prim, halfCircle):
            sdf = star_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
        sdf_matrix.append(sdf)

    sdf_matrix = jnp.stack(sdf_matrix)  # [M, N]
    # 2. 计算每个像素的交点权重
    # 交点权重 = sigmoid(-min_sdf) * sigmoid(second_min_sdf)
    # sorted_sdf = jnp.sort(sdf_matrix, axis=0)  # 沿图元维度排序
    # 添加微小动态噪声（强制 XLA 延迟编译）
    dynamic_noise = jax.random.normal(jax.random.PRNGKey(0), shape=sdf_matrix.shape) * 1e-8
    sdf_matrix = sdf_matrix + dynamic_noise

    # 使用 jax.lax.sort 替代 jnp.sort（显式控制编译行为）
    sorted_sdf = jax.lax.sort(sdf_matrix, dimension=0)  # 沿图元维度排序
    min_sdf = sorted_sdf[0]  # 最近图元的SDF
    second_min_sdf = sorted_sdf[1]  # 第二近图元的SDF

    # 计算交点区域权重（两个图元都接近边界）
    intersection_weight = jax.nn.sigmoid(-min_sdf * 100) * jax.nn.sigmoid(-second_min_sdf * 50)

    # 4. 计算交点区域的MSE损失
    diff = (rendered_rgb.reshape(-1, 3) - target_rgb.reshape(-1, 3)) ** 2
    weighted_diff = diff * intersection_weight[:, None]
    intersection_loss = jnp.mean(weighted_diff)

    # 5. 添加正则化项 - 鼓励图元在交点处平滑过渡
    sdf_diff = jnp.abs(min_sdf - second_min_sdf)
    smoothness_reg = jnp.mean(jnp.exp(-sdf_diff * 100))

    return intersection_loss + 0.1 * smoothness_reg


def create_animation_gif(image_dir, output_file="animation.gif", frame_duration=100):
    """从渲染帧创建GIF动画"""
    frames = []
    frame_files = sorted([f for f in os.listdir(image_dir) if f.startswith("frame_")])
    for frame_file in frame_files:
        img = Image.open(os.path.join(image_dir, frame_file))
        frames.append(np.array(img))
    imageio.mimsave(os.path.join(image_dir, output_file), frames, duration=frame_duration, loop=0)
    print(f"Animation saved to {output_file}")


def create_animation(image_dir, fps=10):
    """
    从渲染帧创建MP4视频动画
    使用FFmpeg生成高质量视频
    """
    image_paths = natsorted(glob(os.path.join(image_dir, '*.png')))
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可改为 'XVID'
    video = cv2.VideoWriter(os.path.join(image_dir, 'process_visualize.mp4'), fourcc, 60, (width, height))
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        video.write(frame)
    video.release()
    print("视频保存成功!")


def save_frame(primitives, grid, output_dir, step):
    """保存优化过程中的帧"""
    rendered = differentiable_rasterize(primitives, grid)
    img = np.array(rendered * 255).astype(np.uint8)
    frame_path = output_dir / f"frame_{step:04d}.png"
    Image.fromarray(img).save(frame_path)


def staged_optimization(primitives, grid, target_img, output_dir):
    """两阶段优化：先优化颜色，再联合优化"""

    print("=== 阶段1: 几何参数优化===")
    second_optimized = optimize_primitives(
        primitives, grid, target_img, output_dir,
        steps=500, lr=0.001
    )
    print("=== 阶段2: 交点优化 ===")
    fully_optimized = optimize_edge(
        second_optimized, grid, target_img, output_dir,
        steps=200, lr=0.01
    )

    return fully_optimized


def to_params(primitives):
    """将图元转换为可优化参数"""
    params, params_type = [], []
    for prim in primitives:
        # 对颜色应用logit变换确保在(0,1)范围内
        color_clipped = jnp.clip(prim.color, 0.01, 0.99)
        color_logit = jnp.log(color_clipped / (1 - color_clipped))
        if isinstance(prim, Circle):
            params.append({
                'center': prim.center,
                'radius': prim.radius,
                'color': color_logit,
                'depth': prim.depth
            })
            params_type.append('0')
        elif isinstance(prim, Rectangle):

            round_log = prim.round
            params.append({
                'center': prim.center,
                'size': prim.size,
                'rotate_theta': prim.rotate_theta,
                'round': round_log,
                'color': color_logit,
                'depth': prim.depth,
            })
            params_type.append('1')
        elif isinstance(prim, Triangle):
            round_log = prim.round

            params.append({
                'center': prim.center,
                'radius': prim.radius,
                'rotate_theta': prim.rotate_theta,
                'round': round_log,
                'color': color_logit,
                'depth': prim.depth,
            })
            params_type.append('2')
        elif isinstance(prim, Capsule):
            # a=jnp.maximum(prim.a,1e-6)
            # a = jnp.maximum(prim.a, 1e-12)  # 对数变换，确保参数空间无约束
            round_log = jnp.log(jnp.maximum(prim.round, 0))
            # params.append({
            #     'center': prim.center,
            #     'radius': radius_log,
            #     'rotate_theta': prim.rotate_theta,
            #     'round': round_log,
            #     'color': color_logit,
            #     'depth': prim.depth
            # })
            params.append({
                'center': prim.center,
                'length': prim.length,
                'a': prim.a,
                'rotate_theta': prim.rotate_theta,
                'round': round_log,
                'color': color_logit,
                'depth': prim.depth
            })
            params_type.append('3')
        elif isinstance(prim, Arc):
            params.append({
                'center': prim.center,
                'shape_theta': prim.shape_theta,
                'rotate_theta': prim.rotate_theta,
                'radius': prim.radius,
                'round': prim.round,
                'color': color_logit,
                'depth': prim.depth
            })
            params_type.append('4')
        elif isinstance(prim, Trapezoid):
            width1 = prim.width1
            width2 = prim.width2
            params.append({
                'center': prim.center,
                'width1': width1,
                'width2': width2,
                'height': prim.height,
                'rotate_theta': prim.rotate_theta,
                'round': prim.round,
                'color': color_logit,
                'depth': prim.depth
            })
            params_type.append('5')
        elif isinstance(prim, Star):

            params.append({
                'center': prim.center,
                'radius': prim.radius,
                'theta': prim.theta,
                'external_angle': prim.external_angle,
                'round': prim.round,
                'k': prim.k,
                'color': color_logit,
                'depth': prim.depth
            })
            params_type.append('6')
        elif isinstance(prim, halfCircle):
            params.append({
                'center': prim.center,
                'radius': prim.radius,
                'rotate_theta': prim.rotate_theta,
                'round': prim.round,
                'color': color_logit,
                'depth': prim.depth
            })
            params_type.append('7')

    return params, params_type


def from_params(params, params_type):
    """将参数转换回图元"""
    prims = []
    for param, type_name in zip(params, params_type):
        # 应用sigmoid确保颜色在(0,1)范围内
        color = 1 / (1 + jnp.exp(-param['color']))

        if type_name == '0':  # 圆形
            prims.append(Circle(
                param['center'],
                param['radius'],
                color,
                param['depth']
            ))
        elif type_name == '1':  # 矩形
            round = param['round']
            round = jnp.minimum(round, jnp.min(param['size']) / 2)

            prims.append(Rectangle(
                param['center'],
                param['size'],
                param['rotate_theta'],
                round,
                color,
                param['depth']
            ))
        elif type_name == '2':  # 三角形
            round = param['round']
            depth = param['depth']

            prims.append(Triangle(
                param['center'],
                param['radius'],
                param['rotate_theta'],
                round,
                color,
                depth
            ))
        elif type_name == '3':
            # a = jnp.exp(param['a'])  # 指数变换，确保为正
            round = jnp.exp(param['round'])
            # a = jnp.exp(param['a'])  # 指数函数确保a始终为正
            # a = jnp.clip(param['a'], 0, 10.0)
            # prims.append(Capsule(
            #     param['center'],
            #     radius,
            #     param['rotate_theta'],
            #     round,
            #     color,
            #     param['depth']
            # ))
            prims.append(Capsule(
                param['center'],
                param['length'],
                param['a'],
                param['rotate_theta'],
                round,
                color,
                param['depth']
            ))
        elif type_name == '4':
            prims.append(Arc(
                param['center'],
                param['shape_theta'],
                param['rotate_theta'],
                param['radius'],
                param['round'],
                color,
                param['depth']
            ))
        elif type_name == '5':
            width1 = param['width1']

            width2 = param['width2']
            prims.append(Trapezoid(
                param['center'],
                width1,
                width2,
                param['height'],
                param['rotate_theta'],
                param['round'],
                color,
                param['depth']
            ))
        elif type_name == '6':
            round = jnp.maximum(param['round'], 0.0)  # 指数变换（实数→正值）
            k = jnp.maximum(param['k'], 0.001)
            prims.append(Star(
                param['center'],
                param['radius'],
                param['theta'],
                param['external_angle'],
                round,
                k,
                color,
                param['depth']
            ))
        elif type_name == '7':
            prims.append(halfCircle(
                param['center'],
                param['radius'],
                param['rotate_theta'],
                param['round'],
                color,
                param['depth']
            ))
    return prims


def create_optimizer(lr, decay_rate=0.95, decay_steps=100):
    # 定义学习率调度器（指数衰减）
    learning_rate_schedule = optax.exponential_decay(
        init_value=lr,  # 初始学习率
        transition_steps=decay_steps,  # 每多少步衰减一次
        decay_rate=decay_rate,  # 衰减率（如0.99表示每次衰减1%）
        staircase=False  # 是否阶梯式衰减（False为连续衰减）
    )

    # 组合优化器：梯度裁剪 + 带学习率衰减的Adam
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # 梯度裁剪
        optax.adam(learning_rate=learning_rate_schedule)  # 使用调度器
    )
    return optimizer


# ==============================
# 优化函数 - 添加梯度裁剪和参数约束
# ==============================
def optimize_primitives(primitives, grid, target_img, output, steps, lr):
    """优化图元参数匹配目标图像"""
    params, params_type = to_params(primitives)

    def run_single_optimization(params, params_type, grid, target_img, output, steps, lr, softness):

        optimization_frames_file = os.path.join(output, "optimization_frames")
        frame_count = 0

        # 渲染函数
        def render(prims):
            return differentiable_rasterize(prims, grid, softness)

        # 初始化优化器 - 添加梯度裁剪
        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(1.0),  # 梯度裁剪
        #     optax.adam(lr)
        # )
        optimizer = create_optimizer(lr=lr, decay_rate=0.99, decay_steps=100)
        opt_state = optimizer.init(params)

        @jit
        def loss_fn(params):
            prims = from_params(params, params_type)
            rendered = differentiable_rasterize(prims, grid, softness)
            # 只比较RGB通道
            target_rgb = target_img[..., :3]
            rendered_rgb = rendered[..., :3]
            diff = rendered_rgb - target_rgb
            mse_loss = jnp.mean(diff ** 2)
            # 添加边缘一致性损失
            edge_loss = edge_consistency_loss(rendered_rgb, target_rgb)

            return mse_loss + edge_loss

        @jit
        def update(params, opt_state):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)

            def freeze_geometry_grad(grad_dict, prim_type):
                frozen_grad = {}
                for k, v in grad_dict.items():
                    if k in ['color', 'depth']:
                        frozen_grad[k] = jnp.zeros_like(v)
                    # # 弯曲胶囊的 'a' 参数梯度缩放
                    elif prim_type == '3' and k == 'a':
                        frozen_grad[k] = v * 1e-5  # 同样缩小梯度
                    elif prim_type == '3' and k == 'round':
                        frozen_grad[k] = jnp.zeros_like(v)

                    else:
                        frozen_grad[k] = v

                return frozen_grad

            grads = [freeze_geometry_grad(grads[i], params_type[i]) for i in range(len(grads))]

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val, grads

        frame_count = 0
        init_img = np.array(render(primitives)) * 255
        init_img = init_img.astype(np.uint8)
        optimization_frames_file = os.path.join(output, "optimization_frames")
        os.makedirs(optimization_frames_file, exist_ok=True)
        Image.fromarray(init_img).save(os.path.join(optimization_frames_file, f"frame_{frame_count:04d}.png"))

        # 优化循环
        losses = []
        grad_history = []  # 用于记录梯度历史

        print("开始优化...")
        success = True
        for step in range(steps):

            prev_params = params.copy()
            params, opt_state, loss_val, grads = update(params, opt_state)
            # print('params:', params)
            losses.append(loss_val)
            grads = jax.tree.map(
                lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), jnp.zeros_like(g), g),
                grads
            )
            # 检查损失是否为NaN
            if jnp.isnan(loss_val):
                print(f"Step {step} 检测到NaN值，恢复上一轮参数")
                success = False
                return success, from_params(prev_params, params_type)

            for i, param in enumerate(params):
                for k, v in param.items():
                    if jnp.any(jnp.isnan(v)) or jnp.any(jnp.isinf(v)):
                        success = False
                        return success, from_params(prev_params, params_type)

            if step % 100 == 0 or step == steps - 1:
                print(f"Step {step}, Loss: {loss_val:.6f}, total grads: {optax.global_norm(grads):.6f}")

                current_prims = from_params(params, params_type)

                current_img = np.array(render(current_prims) * 255)
                current_img = current_img.astype(np.uint8)
                Image.fromarray(current_img).save(
                    os.path.join(optimization_frames_file, f"frame_{frame_count:04d}.png"))
                frame_count += 1
                step_grads = []
                # 新增：遍历每个图元的梯度
                for i, prim_grad in enumerate(grads):
                    grad_info = {}
                    print(f"图元 {i}:")
                    for param_name, g in prim_grad.items():
                        # 计算梯度的L2范数和最大值
                        g_flat = jnp.ravel(g)
                        grad_norm = jnp.linalg.norm(g_flat)
                        grad_max = jnp.max(jnp.abs(g_flat))

                        # 记录梯度信息
                        grad_info[param_name] = {
                            'norm': jax.device_get(grad_norm),
                            'max': jax.device_get(grad_max)
                        }
                    print(f"grad_info: {grad_info}")
                    step_grads.append(grad_info)
                grad_history.append(step_grads)
                # 检查NaN值
                if jnp.isnan(loss_val):
                    print("检测到NaN值，终止优化")
                    for i, param in enumerate(params):
                        for k, v in param.items():
                            if jnp.any(jnp.isnan(v)):
                                print(f"参数 {i} 的 {k} 包含NaN: {v}")
                                print(f"参数 {i} 的 {k} 包含NaN: {v}")
                            break
        # 绘制损失曲线
        plt.figure()
        plt.plot(losses)
        plt.title("Optimization Loss")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.savefig(os.path.join(optimization_frames_file, "optimization_loss.png"))

        optimized_primitives = from_params(params, params_type)

        # 保存优化后的参数
        optimized_params_path = os.path.join(optimization_frames_file, "optimized_params.txt")
        save_parameters(str(optimized_params_path), optimized_primitives, target_img.shape[0], target_img.shape[1])
        print(f"优化后的参数已保存至: {optimized_params_path}")

        # # 创建动画
        # create_animation_gif(optimization_frames_file)
        # create_animation(optimization_frames_file)

        print("优化完成!")
        return success, optimized_primitives

    softness_schedule = [150, 100]  # 逐步降低softness
    for softness in softness_schedule:
        print('softness:', softness)
        success, result = run_single_optimization(
            params, params_type, grid, target_img, output, steps, lr, softness
        )
        if success:
            return result
    return from_params(params, params_type)


def optimize_edge(primitives, grid, target_img, output, steps=500, lr=0.01):
    # 渲染函数
    def render(prims):
        return differentiable_rasterize(prims, grid)

    params, params_type = to_params(primitives)

    # 初始化优化器 - 添加梯度裁剪
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(1.0),  # 梯度裁剪
    #     optax.adam(lr)
    # )
    optimizer = create_optimizer(lr=lr, decay_rate=0.99, decay_steps=100)
    opt_state = optimizer.init(params)

    @jit
    def loss_fn(params):
        prims = from_params(params, params_type)
        rendered = render(prims)
        # 只比较RGB通道
        target_rgb = target_img[..., :3]
        rendered_rgb = rendered[..., :3]
        diff = rendered_rgb - target_rgb
        mse_loss = jnp.mean(diff ** 2)
        # 添加边缘一致性损失
        edge_loss = edge_consistency_loss(rendered_rgb, target_rgb)
        inter_loss = intersection_aware_loss(primitives, grid, target_rgb, rendered_rgb)
        return mse_loss + edge_loss + inter_loss

    @jit
    def update(params, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)

        # 冻结颜色和深度梯度
        def freeze_geometry_grad(grad_dict, prim_type):
            frozen_grad = {}
            for k, v in grad_dict.items():
                if k in ['color', 'depth']:
                    frozen_grad[k] = jnp.zeros_like(v)
                # # 弯曲胶囊的 'a' 参数梯度缩放
                elif prim_type == '3' and k == 'a':
                    frozen_grad[k] = v * 1e-7  # 同样缩小梯度
                elif prim_type == '3' and k == 'round':
                    frozen_grad[k] = v * 1e-8

                else:
                    frozen_grad[k] = v
            return frozen_grad

        grads = [freeze_geometry_grad(grads[i], params_type[i]) for i in range(len(grads))]
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, grads

    frame_count = 0
    init_img = np.array(render(primitives)) * 255
    init_img = init_img.astype(np.uint8)
    optimization_frames_file = os.path.join(output, "edge_optimization_frames")
    os.makedirs(optimization_frames_file, exist_ok=True)
    Image.fromarray(init_img).save(os.path.join(optimization_frames_file, f"frame_{frame_count:04d}.png"))

    # 优化循环
    losses = []
    grad_history = []  # 用于记录梯度历史

    print("开始优化...")
    for step in range(steps):
        prev_params = params.copy()
        params, opt_state, loss_val, grads = update(params, opt_state)
        # print('params:', params)
        losses.append(loss_val)
        grads = jax.tree.map(
            lambda g: jnp.where(jnp.isnan(g) | jnp.isinf(g), jnp.zeros_like(g), g),
            grads
        )
        # 检查损失是否为NaN
        if jnp.isnan(loss_val):
            print(f"Step {step} 检测到NaN值，恢复上一轮参数")

            return from_params(prev_params, params_type)
        # 检查参数是否包含NaN

        for i, param in enumerate(params):
            for k, v in param.items():
                if jnp.any(jnp.isnan(v)) or jnp.any(jnp.isinf(v)):
                    return from_params(prev_params, params_type)

        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step}, Loss: {loss_val:.6f}, total grads: {optax.global_norm(grads):.6f}")

            current_prims = from_params(params, params_type)
            current_img = np.array(render(current_prims) * 255)
            current_img = current_img.astype(np.uint8)
            Image.fromarray(current_img).save(os.path.join(optimization_frames_file, f"frame_{frame_count:04d}.png"))
            frame_count += 1
            step_grads = []
            # 新增：遍历每个图元的梯度
            for i, prim_grad in enumerate(grads):
                grad_info = {}
                # print(f"图元 {i}:")
                for param_name, g in prim_grad.items():
                    # 计算梯度的L2范数和最大值
                    g_flat = jnp.ravel(g)
                    grad_norm = jnp.linalg.norm(g_flat)
                    grad_max = jnp.max(jnp.abs(g_flat))

                    # 记录梯度信息
                    grad_info[param_name] = {
                        'norm': jax.device_get(grad_norm),
                        'max': jax.device_get(grad_max)
                    }
                # print(f"grad_info: {grad_info}")
                step_grads.append(grad_info)
            grad_history.append(step_grads)
            # 检查NaN值
            if jnp.isnan(loss_val):
                print("检测到NaN值，终止优化")
                for i, param in enumerate(params):
                    for k, v in param.items():
                        if jnp.any(jnp.isnan(v)):
                            print(f"参数 {i} 的 {k} 包含NaN: {v}")
                            print(f"参数 {i} 的 {k} 包含NaN: {v}")
                        break

    # 绘制损失曲线
    plt.figure()
    plt.plot(losses)
    plt.title("Optimization Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.savefig(os.path.join(optimization_frames_file, "optimization_loss.png"))
    optimized_primitives = from_params(params, params_type)

    # 保存优化后的参数
    optimized_params_path = os.path.join(optimization_frames_file, "optimized_params.txt")
    save_parameters(str(optimized_params_path), optimized_primitives, target_img.shape[0], target_img.shape[1])
    print(f"优化后的参数已保存至: {optimized_params_path}")

    # # 创建动画
    # create_animation_gif(optimization_frames_file)
    # create_animation(optimization_frames_file)

    print("优化完成!")
    return optimized_primitives


def optimized_differentiable_rasterize(primitives, grid, softness=100, threshold=0.7):
    """
    可微光栅化函数（针对不透明图元，使用画家算法硬覆盖）

    参数:
        primitives: 图元列表（所有图元alpha=1.0）
        grid: 像素网格坐标 (H, W, 2) - 归一化坐标系
        softness: 软化参数控制SDF梯度
        threshold: RGB阈值（接近0设为0，接近1设为1）
    返回:
        rgba图像 (H, W, 4)
    """
    H, W, _ = grid.shape
    points = grid.reshape(-1, 2)  # 展平像素坐标 [N, 2]，N=H*W
    num_pixels = points.shape[0]
    colors = jnp.array([p.color for p in primitives])  # [M, 4]，其中alpha=1.0

    base_colors = colors[:, :3]  # RGB [M, 3]

    # 3. 初始化画布（背景为黑色，可根据需求修改）
    output_rgb = jnp.zeros((num_pixels, 3))  # [N, 3]
    # 记录每个像素是否已被更近的图元覆盖（0=未覆盖，1=已覆盖）
    covered_mask = jnp.zeros(num_pixels, dtype=jnp.bool_)  # [N]

    # 4. 按深度顺序绘制图元（从远到近）
    for i in range(len(primitives)):
        prim = primitives[i]
        # 计算当前图元的SDF（判断像素是否在图元内部）
        if isinstance(prim, Circle):
            sdf = Circle_sdf(prim.center, prim.radius, points)
        elif isinstance(prim, Rectangle):
            sdf = rectangle_sdf(prim.center, prim.size, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Triangle):
            sdf = triangle_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Capsule):
            sdf = curved_capsule_sdf(prim.center, prim.length, prim.a, prim.rotate_theta, prim.round, points)
        elif isinstance(prim, Arc):
            sdf = arc_sdf(prim.center, prim.shape_theta, prim.rotate_theta, prim.radius, prim.round, points)
        elif isinstance(prim, Trapezoid):
            sdf = trapezoid_sdf(prim.center, prim.width1, prim.width2, prim.height, prim.rotate_theta, prim.round,
                                points)
        elif isinstance(prim, Star):
            sdf = star_sdf(prim.center, prim.radius, prim.theta, prim.external_angle, prim.round, prim.k, points)
        elif isinstance(prim, halfCircle):
            sdf = star_sdf(prim.center, prim.radius, prim.rotate_theta, prim.round, points)
        # SDF <= 0 表示像素在图元内部（不透明，完全覆盖）
        sdf_clamped = jnp.clip(sdf, -50.0, 50.0)

        inside_prim = sdf_clamped <= 0  # [N]，True=像素在图元内

        # # 仅覆盖未被更近图元占据的像素（未覆盖区域）
        # # 逻辑：当前像素在图元内，且未被之前的图元覆盖
        update_mask = inside_prim  # [N]
        #
        # # 对图元颜色进行阈值处理（接近0设为0，接近1设为1）
        prim_rgb = base_colors[i]  # [3]
        processed_rgb = jnp.where(
            prim_rgb > threshold,  # 接近0
            1.0,
            0.0
        )

        # 更新画布：仅在update_mask为True的位置填充当前图元颜色
        output_rgb = jnp.where(
            update_mask[:, jnp.newaxis],  # 扩展为[N, 1]匹配RGB维度
            processed_rgb,  # 当前图元颜色
            output_rgb  # 保持原有颜色（未覆盖区域不变）
        )
        covered_mask = covered_mask | update_mask
    # 5. 组合RGBA（不透明，alpha固定为1.0）
    output_alpha = jnp.ones(num_pixels)  # [N]
    rgba = jnp.hstack([output_rgb, output_alpha[:, jnp.newaxis]])  # [N, 4]

    # 重塑为图像
    rgba_image = rgba.reshape(H, W, 4)

    return rgba_image


# ==============================
# 主函数
# ==============================
def main(target_image_path, init_params_path, output_dir):
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    # 读取目标图像
    target_img = iio.imread(target_image_path).astype(jnp.float32) / 255.0
    if jnp.any(jnp.isnan(target_img)):
        raise ValueError("目标图像包含NaN值")

    # 处理图像通道
    if target_img.ndim == 2:  # 灰度图转RGB
        target_img = jnp.stack([target_img] * 3, axis=-1)
    elif target_img.shape[2] == 4:  # 保留Alpha通道
        pass
    else:  # 只有RGB通道，添加Alpha通道
        alpha = jnp.ones(target_img.shape[:2] + (1,))
        target_img = jnp.concatenate([target_img, alpha], axis=-1)

    height, width, _ = target_img.shape
    print(f"目标图像尺寸: {width}x{height} 像素")

    # 创建归一化网格
    grid = create_normalized_grid(height, width)

    # 检查网格是否包含NaN
    if jnp.any(jnp.isnan(grid)):
        raise ValueError("网格坐标系包含NaN值")

    # 加载初始参数
    primitives = load_parameters(init_params_path)
    print(f"加载了 {len(primitives)} 个图元")

    # 验证初始参数
    for i, prim in enumerate(primitives):
        if jnp.any(jnp.isnan(prim.center)) or jnp.any(jnp.isnan(prim.color)):
            print(f"警告: 图元 {i} 包含NaN值")
        if isinstance(prim, Circle) and prim.radius < 0.01:
            print(f"警告: 图元 {i} 半径过小 ({prim.radius})")

    # 渲染函数
    def render(prims):
        prims = sorted(prims, key=lambda p: p.depth)
        return differentiable_rasterize(prims, grid)

    initial_params_path = output_dir / "initial_params.txt"
    save_parameters(str(initial_params_path), primitives, height, width)

    # 保存初始渲染图像
    initial_img = render(primitives)
    initial_img_path = output_dir / "initial_image.png"
    iio.imwrite(str(initial_img_path), (initial_img * 255).astype(np.uint8))
    print(f"初始渲染图像已保存至: {initial_img_path}")

    # 使用两阶段优化
    optimized_primitives = staged_optimization(
        primitives, grid, target_img, output_dir
    )
    print(f"优化用时：{time.time() - start_time}")

    optimized_params_path = output_dir / "optimized_params.txt"
    save_parameters(str(optimized_params_path), optimized_primitives, height, width)
    print(f"优化后的参数已保存至: {optimized_params_path}")

    optimized_primitives = sorted(optimized_primitives, key=lambda p: p.depth)

    optimized_img = optimized_differentiable_rasterize(optimized_primitives, grid, softness=100.0)
    optimized_img_path = output_dir / "optimized_image.png"
    iio.imwrite(str(optimized_img_path), (optimized_img * 255).astype(np.uint8))
    print(f"优化后的图像已保存至: {optimized_img_path}")

    # 可视化结果
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title("Target Image")
    plt.imshow(np.array(target_img[..., :3]))

    plt.subplot(132)
    plt.title("Initial Rendering")
    plt.imshow(np.array(initial_img[..., :3]))

    plt.subplot(133)
    plt.title("Optimized Rendering")
    plt.imshow(np.array(optimized_img[..., :3]))

    plt.tight_layout()
    plt.savefig(str(output_dir / "comparison.png"), dpi=150)
    print(f"对比图像已保存至: {output_dir / 'comparison.png'}")


if __name__ == "__main__":
    name = "star"
    target_image_path = "images/" + name + ".png"
    target_params_path = "infos/" + name + ".json"
    output = "output/20250820/" + name
    main(target_image_path, target_params_path, output)

