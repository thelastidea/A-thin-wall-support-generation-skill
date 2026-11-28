# -*- coding: utf-8 -*-
# Copyright (C) 2026 thelastidea — SPDX-License-Identifier: AGPL-3.0-only
"""薄墙件的预览图（纯 numpy + PIL + matplotlib，无需 GL / scipy）

可选工具：技能默认只导 STL、不自动渲染；要看形状/遮挡时才显式运行本脚本。

用法：
    python scripts/preview.py part.stl --out <目录> --normal y --up z [--zoom "cx,cy,cz,size"]
    python scripts/preview.py part.stl --out <目录> --iso [--el 115 --az -90]

出四张 PNG（全部是正投影 / 剖切，遮挡由逐像素 z-buffer 判定，不是画家算法）：
    _1_whole_front.png  整件正视（沿墙法向，看窗口位置与膜的存在）
    _2_corner_front.png 角部正视（默认 26mm 视域：量缝宽、数连接件）
    _3_mid_section.png  中层剖（沿墙厚中面切开）：膜边界、缝宽、柱脚
    _4_gap_section.png  跨缝剖（含墙厚）：膜是否居中、楔形最宽是否 ≤ 墙厚 − 线宽
加 --iso 再出一张 _5_iso.png 整件斜视（--el/--az 调角度，默认 115/-90：
115 = 从法向再往上抬 25°，能同时看到顶面与墙厚；el<90 是从下方看）。
量尺寸以 1:1 正投影与剖切为准；斜视只回答"整件长什么样"，不要再另写
matplotlib Poly3DCollection 出图——那是按面平均深度排序，5mm 厚壁上的
2.5mm 错位会翻面，画出来像零件碎了。

剖切只用 trimesh.intersections.mesh_plane（不需要 scipy / slice_mesh_plane）。
"""
from __future__ import annotations

import argparse
import inspect
import os

import numpy as np
import trimesh
import trimesh.intersections as TI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

AX = {"x": 0, "y": 1, "z": 2}
_AXNAME = ("X 向", "Y 向", "Z 向")


# ---------------- 基本框架 ----------------

def _frame(normal_axis="y", up_axis="z"):
    """返回 (h, u, n)：水平、向上、墙法向在世界坐标里的轴下标"""
    n = AX[normal_axis]
    u = AX[up_axis]
    h = ({0, 1, 2} - {n, u}).pop()
    return h, u, n


def _basis(az, el):
    """视平面基：right / up / eye（单位向量，坐标在 (h, u, n) 空间）

    el 是相对**零件平面**的仰角（0 = 贴着面看，90 = 正对法向）
    """
    a, e = np.radians(az), np.radians(el)
    eye = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    upv = np.cross(eye, right)
    return right, upv, eye


def _auto_zoom(mesh, h, u, n, size=26.0):
    """定位口的「上左角」：楔形是唯一斜面，用它的包围盒取 up 边与左边界"""
    nf = np.abs(np.array(mesh.face_normals, dtype=float, copy=True))
    ob = nf.max(axis=1) < 0.9
    V = np.asarray(mesh.vertices, dtype=float)
    if ob.sum() < 8:
        b = np.asarray(mesh.bounds)
        return {"center": b.mean(axis=0), "size": float(size),
                "up": b[1, u], "left": b[0, h]}
    pts = V[np.asarray(mesh.faces)[ob]].reshape(-1, 3)
    c = pts.mean(axis=0)
    up_edge, left_edge = pts[:, u].max(), pts[:, h].min()
    c[h] = left_edge + size / 2.0          # 横向落在左边界往里半个视野
    c[u] = up_edge - size / 2.0            # 纵向让 up 边贴住视野上缘
    return {"center": c, "size": float(size), "up": up_edge, "left": left_edge}


# ---------------- 深度缓冲光栅化 ----------------

def _raster(mesh, h, u, n, el, az, crop=None, px=1600):
    V = np.asarray(mesh.vertices, dtype=float)[:, [h, u, n]]
    F = np.asarray(mesh.faces)
    Nw = np.array(mesh.face_normals, dtype=float, copy=True)[:, [h, u, n]]
    right, upv, eye = _basis(az, el)
    X, Y, D = V @ right, V @ upv, V @ eye

    # 视野：只用裁切框（8 个角点投影）定框，**不筛三角形**
    # （布尔后的大三角形质心可能在框里、顶点却横跨整块板，按质心筛会把视野撑回全宽）
    if crop is not None:
        c, size = crop
        # c2 按**排列后**的顺序 (h, u, n) 取值；下面 q[0]/q[1]/q[2] 也是这个顺序
        c2 = np.array([c[h], c[u], c[n]], dtype=float)
        n_rng = max(3.0, c2[2])
        corners = []
        for dh in (-0.5, 0.5):
            for du in (-0.5, 0.5):
                for dn in (-0.5, 0.5):
                    q = c2.copy()
                    q[0] += dh * size
                    q[1] += du * size
                    q[2] += dn * n_rng
                    corners.append(q)
        corners = np.array(corners)
        cx, cy = (corners @ right).mean(), (corners @ upv).mean()
        half = max(np.ptp(corners @ right), np.ptp(corners @ upv)) / 2.0 * 1.06
    else:
        cx, cy = (X.min() + X.max()) / 2.0, (Y.min() + Y.max()) / 2.0
        half = max(X.max() - X.min(), Y.max() - Y.min()) / 2.0 * 1.04
    sc = (px - 1) / (2.0 * half)
    px_x = (X - cx) * sc + px / 2.0
    px_y = px / 2.0 - (Y - cy) * sc

    zbuf = np.full((px, px), -1e18)
    img = np.ones((px, px, 3))
    nbuf = np.zeros((px, px, 3))

    # 光照（视空间）：左上偏前
    L = np.array([-0.38, 0.52, 0.76])
    L /= np.linalg.norm(L)
    nx, ny, nz = Nw @ right, Nw @ upv, Nw @ eye
    inten = np.clip(np.abs(nx * L[0] + ny * L[1] + nz * L[2]), 0, 1)
    # 压低动态范围：过曝会把受光面画成纯白，看着像漏画的洞
    base = 0.46 + 0.42 * inten
    # 景深色调：面越靠近相机越亮，膜/面板/柱脚才不会糊成一片
    dd = D[F].mean(axis=1)
    rng = max(float(dd.max() - dd.min()), 1e-6)
    base = np.clip(base + 0.10 * (dd - dd.min()) / rng - 0.05, 0.30, 0.92)
    cols = np.stack([base * 0.95, base * 0.96, base * 1.0], axis=1)

    for i in range(len(F)):
        fx, fy, fd = px_x[F[i]], px_y[F[i]], D[F[i]]
        x0, x1 = int(np.floor(fx.min())), int(np.ceil(fx.max()))
        y0, y1 = int(np.floor(fy.min())), int(np.ceil(fy.max()))
        x0, x1 = max(x0, 0), min(x1, px - 1)
        y0, y1 = max(y0, 0), min(y1, px - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        # 边函数（兼容两种绕序）
        e0 = (fx[1] - fx[0]) * (gy - fy[0]) - (fy[1] - fy[0]) * (gx - fx[0])
        e1 = (fx[2] - fx[1]) * (gy - fy[1]) - (fy[2] - fy[1]) * (gx - fx[1])
        e2 = (fx[0] - fx[2]) * (gy - fy[2]) - (fy[0] - fy[2]) * (gx - fx[2])
        s = np.sign(e0[0, 0] + e1[0, 0] + e2[0, 0])
        if s == 0:
            continue
        inside = (s * e0 >= 0) & (s * e1 >= 0) & (s * e2 >= 0)
        if not inside.any():
            continue
        area = s * (e0 + e1 + e2)
        w0, w1, w2 = s * e0 / area, s * e1 / area, s * e2 / area
        # 注意：e0 是 (P1-P0)×(P-P0)，对应 P2 的对边面积，故权重与顶点是错位对齐的：
        # w0→fd[2]、w1→fd[0]、w2→fd[1]。写反了在正视下无感（三顶点深度相同），
        # 但斜视时插值深度会被拧成歪面，出现整片假遮挡碎块。
        depth = w1 * fd[0] + w2 * fd[1] + w0 * fd[2]
        zz = zbuf[y0:y1 + 1, x0:x1 + 1]
        m = inside & (depth > zz)
        if not m.any():
            continue
        zz[m] = depth[m]
        cc = img[y0:y1 + 1, x0:x1 + 1]
        cc[m] = cols[i]
        nn = nbuf[y0:y1 + 1, x0:x1 + 1]
        nn[m] = (nx[i], ny[i], nz[i])

    # 1) 补缝：布尔后常有 T 型接边，光栅化会在缝上留白点 → 用 3×3 邻域已覆盖像素填 2 轮
    cov = zbuf > -1e17
    for _ in range(2):
        acc_c = np.zeros_like(img)
        acc_n = np.zeros_like(nbuf)
        cnt = np.zeros((px, px))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                v = np.roll(np.roll(cov, dy, axis=0), dx, axis=1)
                acc_c += np.roll(np.roll(img, dy, axis=0), dx, axis=1) * v[..., None]
                acc_n += np.roll(np.roll(nbuf, dy, axis=0), dx, axis=1) * v[..., None]
                cnt += v
        add = (~cov) & (cnt > 0)
        if not add.any():
            break
        img[add] = acc_c[add] / cnt[add][:, None]
        nbuf[add] = acc_n[add] / cnt[add][:, None]
        cov |= add

    # 2) 边线：轮廓（未覆盖相邻）+ 折边（邻像素法向夹角大）+ 深度突变
    edge = np.zeros_like(cov)
    for sh in (1, -1):
        edge |= cov & ~np.roll(cov, sh, axis=0)
        edge |= cov & ~np.roll(cov, sh, axis=1)
    nn = nbuf / np.maximum(np.linalg.norm(nbuf, axis=2, keepdims=True), 1e-9)
    for ax in (0, 1):
        for sh in (1, -1):
            dot = np.sum(nn * np.roll(nn, sh, axis=ax), axis=2)
            edge |= cov & (np.abs(dot) < np.cos(np.radians(35)))
    grad = np.zeros_like(zbuf)
    grad[:-1, :] = np.maximum(grad[:-1, :], np.abs(np.diff(zbuf, axis=0)))
    grad[:, :-1] = np.maximum(grad[:, :-1], np.abs(np.diff(zbuf, axis=1)))
    # 只留“真台阶”：阈值取 1.2 像素对应的深度，避免近平行面产生碎点
    edge |= grad > 1.2 * (2.0 * half / px)
    img[edge] *= 0.55
    img[~cov] = 1.0
    return img, (cx, cy, half)


def render_shaded(mesh, out, h, u, n, el, az, crop=None, title="", px=1500):
    img, (cx, cy, half) = _raster(mesh, h, u, n, el, az, crop=crop, px=px)
    im = np.clip(img, 0, 1)
    hh = (im.shape[0] // 2) * 2
    im = im[:hh, :hh].reshape(hh // 2, 2, hh // 2, 2, 3).mean(axis=(1, 3))
    im = (im * 255).astype(np.uint8)                  # 2× 盒式降采样当抗锯齿
    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=150)
    ax.imshow(im, extent=[-half, half, -half, half], origin="upper")
    ax.set_xlabel("视平面横（相对视野中心，mm）")
    ax.set_ylabel("视平面纵（相对视野中心，mm）")
    ax.grid(True, ls=":", lw=0.4, color="#c9ced6")
    ax.set_title(title + "｜视野 ±%.1f mm" % half, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------- 二维剖面 ----------------

def _intersect(mesh, origin, normal):
    """trimesh 版本兼容：0.9 是 (mesh, origin, normal)，5.x 改成 (mesh, normal, origin)

    传错顺序不会抛异常，只会把剖面切到别的位置（实测 X=50 切到 X=1.0），所以按签名判。
    """
    params = list(inspect.signature(TI.mesh_plane).parameters)[:3]
    if params[:3] == ["mesh", "plane_origin", "plane_normal"]:
        return TI.mesh_plane(mesh, origin, normal)
    return TI.mesh_plane(mesh, normal, origin)


def section_2d(mesh, normal_idx, origin, out, aa, bb, crop=None, title=""):
    normal = np.zeros(3)
    normal[normal_idx] = 1.0
    seg = _intersect(mesh, np.asarray(origin, dtype=float), normal)
    if seg is None or len(seg) == 0:
        raise ValueError("剖切面无交线，检查 --normal 与剖切位置")
    lines = np.asarray(seg)[:, :, [aa, bb]]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    if crop is not None:
        c, size = crop
        # 只定视野、不裁线段：横跨整个口的长边中点可能落在视野外，
        # 按中点裁会把整条长边丢掉（曾导致剖面空白）
        c2 = np.array([c[aa], c[bb]], dtype=float)
        xlim = (c2[0] - size / 2.0, c2[0] + size / 2.0)
        ylim = (c2[1] - size / 2.0, c2[1] + size / 2.0)
    ax.add_collection(LineCollection(lines, colors="#1b1f26", linewidths=0.7))
    if crop is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("%s (mm)" % _AXNAME[aa])
    ax.set_ylabel("%s (mm)" % _AXNAME[bb])
    ax.grid(True, ls=":", lw=0.4, color="#b9bec7")
    ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------- 主流程 ----------------

def preview(stl, out_dir, normal_axis="y", up_axis="z", zoom=None,
            iso=False, iso_el=115.0, iso_az=-90.0):
    os.makedirs(out_dir, exist_ok=True)
    mesh = trimesh.load(stl, force="mesh")
    h, u, n = _frame(normal_axis, up_axis)
    stem = os.path.splitext(os.path.basename(stl))[0]
    if zoom is None:
        info = _auto_zoom(mesh, h, u, n)
        zc, zs = info["center"], info["size"]
    else:
        vals = [float(v) for v in zoom.split(",")]
        zc, zs = (np.array(vals[:3], dtype=float),
                  float(vals[3]) if len(vals) > 3 else 26.0)
    lo, hi = np.asarray(mesh.bounds)
    out = []

    p = os.path.join(out_dir, stem + "_1_whole_front.png")
    out.append(render_shaded(mesh, p, h, u, n, el=90, az=-90,
                             title="整件正视（沿墙法向）｜看窗口位置与口内的膜"))
    p = os.path.join(out_dir, stem + "_2_corner_front.png")
    out.append(render_shaded(mesh, p, h, u, n, el=90, az=-90, crop=(zc, zs),
                             title="角部正视｜%.0f mm 视域：量缝宽、数连接件" % zs))

    origin = np.zeros(3)
    origin[n] = 0.5 * (lo[n] + hi[n])
    p = os.path.join(out_dir, stem + "_3_mid_section.png")
    out.append(section_2d(mesh, n, origin, p, h, u, crop=(zc, zs),
                          title="中层剖（墙厚中面）｜缝宽按层高分区、受 1.5×线宽托底"))

    origin2 = np.zeros(3)
    origin2[h] = zc[h]
    p = os.path.join(out_dir, stem + "_4_gap_section.png")
    out.append(section_2d(mesh, h, origin2, p, u, n, crop=(zc, zs),
                          title="跨缝剖（含墙厚）｜膜须居中、楔形最宽 ≤ 墙厚 − 线宽"))

    if iso:
        p = os.path.join(out_dir, stem + "_5_iso.png")
        out.append(render_shaded(
            mesh, p, h, u, n, el=iso_el, az=iso_az,
            title="整件斜视 el=%.0f° az=%.0f°｜遮挡逐像素判定，墙能挡住膜"
                  % (iso_el, iso_az)))
    for p in out:
        print("写出", p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--out", default=".")
    ap.add_argument("--normal", default="y", choices=list(AX))
    ap.add_argument("--up", default="z", choices=list(AX))
    ap.add_argument("--zoom", default=None, help='"cx,cy,cz" 或 "cx,cy,cz,size"')
    ap.add_argument("--iso", action="store_true", help="额外出一张整件斜视 _5_iso.png")
    ap.add_argument("--el", type=float, default=None,
                    help="斜视仰角（0=贴面，90=正对墙法向；给值即隐含 --iso）")
    ap.add_argument("--az", type=float, default=None,
                    help="斜视方位角（-90 与正视同侧、左右对称）")
    a = ap.parse_args()
    preview(a.stl, a.out, a.normal, a.up, a.zoom,
            iso=(a.iso or a.el is not None or a.az is not None),
            iso_el=115.0 if a.el is None else a.el,
            iso_az=-90.0 if a.az is None else a.az)
