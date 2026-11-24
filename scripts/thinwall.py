# -*- coding: utf-8 -*-
# Copyright (C) 2026 thelastidea — SPDX-License-Identifier: AGPL-3.0-only
"""镂空口「居中薄墙 + 小方柱」通用工具（FDM 3D 打印）

用法（作为库）：
    import thinwall as tw
    cut, fill = tw.window(axis="y", a_lo=0, a_hi=3, u0=20, u1=170, v0=60, v1=220,
                          r=8, spec=tw.SkinSpec())
    solid = tw.subtract(solid, cut)
    solid = tw.add(solid, fill)
    tw.clean_export(solid, "part.stl")

自带自检：python thinwall.py demo --out <目录>
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix


@dataclass
class SkinSpec:
    skin: float = 0.45      # 薄墙厚
    gap_t: float = 0.4      # 上方缝
    gap_b: float = 0.6      # 下方缝
    gap_s: float = 0.5      # 两端缝
    post: float = 0.7       # 方柱截面（两侧/上边柱宽）
    pitch: float = 7.0      # 方柱间距
    embed: float = 0.4      # 方柱嵌入墙体的深度（必须 > 切口的溢出量）
    corner_inset: float = 0.5  # 膜圆角 = 窗口圆角 − 此值；必须 ≥ gap_s，否则圆角处膜会顶到主体（相切融合）
    top_flare: bool = False    # 上边柱：靠墙端按 top_flare_deg 对称张开到 top_flare_w 厚
    top_flare_deg: float = 30.0
    top_flare_w: float = 2.0   # 张开后的总厚度（断面）
    post_b: float = 1.0        # 下边方柱宽度（厚度仍是 skin）
    post_side: bool = False    # 柱子改到左右两端（跨 gap_s 的左右缝），上/下边不布柱
    flare_clear: float = 0.4   # 张开最宽处 = 墙厚 − 此值（两侧各让 0.2，不顶满墙面）


# ---------- 基础几何 ----------

def _place(axis, a, u, v):
    if axis == "x":
        return [a, u, v]
    if axis == "y":
        return [u, a, v]
    return [u, v, a]


def _extents(axis, a0, a1, u0, u1, v0, v1):
    e = [u1 - u0, v1 - v0, a1 - a0] if axis == "z" else (
        [a1 - a0, u1 - u0, v1 - v0] if axis == "y" else [a1 - a0, u1 - u0, v1 - v0])
    if axis == "y":
        e = [u1 - u0, a1 - a0, v1 - v0]
    return e


def box(axis, a0, a1, u0, u1, v0, v1):
    e = _extents(axis, a0, a1, u0, u1, v0, v1)
    m = trimesh.creation.box(extents=e)
    m.apply_translation(_place(axis, (a0 + a1) / 2.0, (u0 + u1) / 2.0, (v0 + v1) / 2.0))
    return m


def _cyl(axis, a0, a1, u, v, r):
    m = trimesh.creation.cylinder(radius=r, height=a1 - a0, sections=48)
    if axis == "x":
        m.apply_transform(rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(_place(axis, (a0 + a1) / 2.0, u, v))
    return m


def _union(parts):
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for p in parts[1:]:
        out = trimesh.boolean.union([out, p], engine="manifold")
    return out


def _av_prism(axis, u0, u1, prof):
    """(a,v) 平面里的多边形 prof 沿 u 方向挤出（两端封盖）"""
    area = sum(prof[i][0] * prof[(i + 1) % len(prof)][1]
               - prof[(i + 1) % len(prof)][0] * prof[i][1] for i in range(len(prof)))
    if area < 0:
        prof = prof[::-1]
    V = []
    for u in (u0, u1):
        for a, v in prof:
            V.append(_place(axis, a, u, v))
    V = np.array(V, dtype=float)
    n = len(prof)
    faces = [[0, i + 1, i] for i in range(1, n - 1)]
    faces += [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    m = trimesh.Trimesh(V, np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def _au_prism(axis, v0, v1, prof):
    """(a,u) 平面里的多边形 prof 沿 v 方向挤出（两端封盖）"""
    area = sum(prof[i][0] * prof[(i + 1) % len(prof)][1]
               - prof[(i + 1) % len(prof)][0] * prof[i][1] for i in range(len(prof)))
    if area < 0:
        prof = prof[::-1]
    V = []
    for v in (v0, v1):
        for a, u in prof:
            V.append(_place(axis, a, u, v))
    V = np.array(V, dtype=float)
    n = len(prof)
    faces = [[0, i + 1, i] for i in range(1, n - 1)]
    faces += [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    m = trimesh.Trimesh(V, np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def flare_size(spec, wall):
    """返回 (gw, rn)。

    gw = 每侧增长量：最宽处 = min(top_flare_w, 墙厚 − flare_clear)，再减去膜厚取半
         （默认 flare_clear=0.4 → 两侧各留 0.2，不顶满墙面、也不凸出墙外）
    rn = 斜坡在 u/v 方向的长度：**沿用原来 30° 时的长度**（按墙厚封顶、不含 flare_clear），
         所以最宽处变窄后实际坡度会变浅（1.6 墙：0.375 / 0.996 ≈ 20.6°）
    """
    target = min(spec.top_flare_w, wall - spec.flare_clear)
    gw = max(0.0, 0.5 * (target - spec.skin))
    span = 0.5 * (min(spec.top_flare_w, wall) - spec.skin)
    rn = max(0.0, span) / math.tan(math.radians(spec.top_flare_deg)) if gw > 0 else 0.0
    return gw, rn


def post_flare(axis, a0, a1, u0, u1, v_tip, v_edge, grow):
    """柱子靠墙那一端按 30° 对称张开 grow，到墙面 v_edge 为止，不埋进墙"""
    return _av_prism(axis, u0, u1,
                     [(a0, v_tip), (a1, v_tip), (a1 + grow, v_edge), (a0 - grow, v_edge)])


def post_wedge(axis, a0, a1, u0, u1, v_root, v_edge, v_tip, grow,
               a_lo=None, a_hi=None):
    """柱子根部带 45° 斜坡：从膜边 v_tip 到墙面 v_edge 厚度由 (a0..a1) 渐变到
    (a0-grow .. a1+grow)，越过 v_edge 后保持胖截面一直到 v_root（埋在墙里）。
    grow = |v_tip - v_edge| 时正好 45°。"""
    if a_lo is not None:
        grow = min(grow, max(0.0, a0 - a_lo), max(0.0, a_hi - a1))
    s = 1.0 if (v_root - v_edge) >= 0 else -1.0
    prof = [(a0 - grow, v_root), (a1 + grow, v_root),
            (a1 + grow, v_edge), (a1, v_tip), (a0, v_tip), (a0 - grow, v_edge)]
    area = sum(prof[i][0] * prof[(i + 1) % 6][1] - prof[(i + 1) % 6][0] * prof[i][1]
               for i in range(6))
    if area < 0:
        prof = prof[::-1]
    V = []
    for u in (u0, u1):
        for a, v in prof:
            V.append(_place(axis, a, u, v))
    V = np.array(V, dtype=float)
    n = len(prof)
    faces = [[0, i + 1, i] for i in range(1, n - 1)]                 # u0 端盖
    faces += [[n, n + i, n + i + 1] for i in range(1, n - 1)]        # u1 端盖
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    m = trimesh.Trimesh(V, np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def rounded(axis, a0, a1, u0, u1, v0, v1, r):
    """沿 axis 的法向挤出、在 (u,v) 平面内是圆角矩形"""
    if r is None or r <= 0.05:
        return box(axis, a0, a1, u0, u1, v0, v1)
    parts = [box(axis, a0, a1, u0 + r, u1 - r, v0, v1),
             box(axis, a0, a1, u0, u1, v0 + r, v1 - r)]
    parts += [_cyl(axis, a0, a1, u, v, r) for u, v in
              ((u0 + r, v0 + r), (u1 - r, v0 + r), (u0 + r, v1 - r), (u1 - r, v1 - r))]
    return _union(parts)


# ---------- 核心：一扇口的 切 + 填 ----------

def _prism(axis, a0, a1, poly_uv):
    """(u,v) 平面内的多边形沿 a 方向挤出"""
    pts = [(float(u), float(v)) for u, v in poly_uv]
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
               - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area < 0:
        pts = pts[::-1]
    n = len(pts)
    V = np.array([_place(axis, a, u, v) for a in (a0, a1) for (u, v) in pts], dtype=float)
    faces = [[0, i + 1, i] for i in range(1, n - 1)]
    faces += [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    m = trimesh.Trimesh(V, np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def _loft(axis, v0, v1, q0, q1):
    """两端各一个矩形截面（q = (a0,a1,u0,u1)）沿 v 方向放样"""
    def pts(v, q):
        a0, a1, u0, u1 = q
        return [(u0, v, a0), (u1, v, a0), (u1, v, a1), (u0, v, a1)]
    A = pts(v0, q0)
    B = pts(v1, q1)
    V = np.array([_place(axis, a, u, v) for (u, v, a) in A + B], dtype=float)
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([i, j, 4 + j])
        faces.append([i, 4 + j, 4 + i])
    m = trimesh.Trimesh(V, np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def top_fin(axis, m0, m1, u_center, v_top, half0=0.35, half_max=5.0,
            deg=30.0, u_lim=None, thick=1.0):
    """上边的鳍：根部窄（2*half0）、与膜同厚，按 deg 在面内张开到 2*half_max，
    厚度方向同样按 deg 从膜厚加厚到 thick；末端顶到口的上边（不埋进墙）。
    v_top 是口的上边；根部在 v_top - run 处。"""
    t = math.tan(math.radians(deg))
    hm = half_max
    if u_lim is not None:
        hm = min(hm, max(half0, u_center - u_lim[0]), max(half0, u_lim[1] - u_center))
    run = (hm - half0) / t
    v_root = v_top - run
    mc = 0.5 * (m0 + m1)
    mt = 0.5 * max(thick, m1 - m0)
    rt = min((mt - 0.5 * (m1 - m0)) / t, run)          # 厚度张开所需的长度
    hh = half0 + rt * t                                 # 加厚段末端的半宽
    parts = []
    if rt > 1e-6:
        parts.append(_loft(axis, v_root, v_root + rt,
                           (m0, m1, u_center - half0, u_center + half0),
                           (mc - mt, mc + mt, u_center - hh, u_center + hh)))
        parts.append(_prism(axis, mc - mt, mc + mt, [
            (u_center - hh, v_root + rt), (u_center + hh, v_root + rt),
            (u_center + hm, v_top), (u_center - hm, v_top)]))
    else:
        parts.append(_prism(axis, m0, m1, [
            (u_center - half0, v_root), (u_center + half0, v_root),
            (u_center + hm, v_top), (u_center - hm, v_top)]))
    return _union(parts)


def top_fin_run(thick=10.0, root=0.7, deg=30.0):
    return (0.5 * (thick - root)) / math.tan(math.radians(deg))


def window(axis, a_lo, a_hi, u0, u1, v0, v1, r=8.0, spec=None, overshoot=1.0):
    """返回 (cut_parts, fill_parts)。

    axis      : 口所在墙面的法向；a_lo/a_hi 是该方向的墙面两侧
    u0..u1    : 口的第一方向范围；v0..v1 是第二方向范围（gap_t 贴 v1，gap_b 贴 v0）
    overshoot : 切口沿 a 方向溢出墙体的量，保证切透
    """
    spec = spec or SkinSpec()
    cut = [rounded(axis, a_lo - overshoot, a_hi + overshoot, u0, u1, v0, v1, r)]

    m0 = a_lo + (a_hi - a_lo - spec.skin) / 2.0      # 薄墙居中
    m1 = m0 + spec.skin
    rr = max(r - spec.corner_inset, 0.0)
    fill = [rounded(axis, m0, m1, u0 + spec.gap_s, u1 - spec.gap_s,
                    v0 + spec.gap_b, v1 - spec.gap_t, rr)]

    if spec.post_side:
        # 柱子改到口的左右两端：跨 gap_s 的左右缝，沿 v 均布、两端固定
        # 离侧面远的那一端（u1）靠墙端按 top_flare_deg 张开，近的那端保持直条
        vv0 = v0 + spec.gap_b + rr + spec.post / 2.0
        vv1 = v1 - spec.gap_t - rr - spec.post / 2.0
        gw = 0.0
        rn = 0.0
        if spec.top_flare:
            gw, rn = flare_size(spec, a_hi - a_lo)
        if vv1 > vv0 - 1e-6:
            n = max(1, int(round((vv1 - vv0) / spec.pitch)))
            for i in range(n + 1):
                v = vv0 + (vv1 - vv0) * i / n
                fill.append(box(axis, m0, m1, u0, u0 + spec.gap_s,
                                v - spec.post / 2, v + spec.post / 2))
                if gw > 0:
                    fill.append(_au_prism(axis, v - spec.post / 2, v + spec.post / 2,
                                          [(m0, u1 - rn), (m1, u1 - rn),
                                           (m1 + gw, u1), (m0 - gw, u1)]))
                else:
                    fill.append(box(axis, m0, m1, u1 - spec.gap_s, u1,
                                    v - spec.post / 2, v + spec.post / 2))
        return cut, fill

    # 方柱只在口的上/下两条边跨缝（沿 v 方向：墙里 0.4 → 跨缝 → 到膜边）
    # 最外侧两根的**外沿与膜边平齐**（不伸到口边/主体上），中间按 pitch 均分
    u_lo = u0 + spec.gap_s + rr + max(spec.post, spec.post_b) / 2.0
    u_hi = u1 - spec.gap_s - rr - max(spec.post, spec.post_b) / 2.0
    if u_hi > u_lo - 1e-6:
        n = max(1, int(round((u_hi - u_lo) / spec.pitch)))
        for i in range(n + 1):
            u = u_lo + (u_hi - u_lo) * i / n
            # 下边：普通方柱（post_b × skin 直条，无斜坡），跨缝到墙面为止，不埋进墙
            fill.append(box(axis, m0, m1, u - spec.post_b / 2, u + spec.post_b / 2,
                            v0, v0 + spec.gap_b))
            if spec.top_flare:
                # 上边：靠墙端张开到 top_flare_w 厚，到墙面为止、不埋进墙
                # 最宽处 = min(目标厚度, 墙厚 − flare_clear)；斜坡长度沿用原 30° 的长度
                gw, rn = flare_size(spec, a_hi - a_lo)
                fill.append(post_flare(axis, m0, m1, u - spec.post / 2, u + spec.post / 2,
                                       v1 - rn, v1, gw))
            else:
                fill.append(post_wedge(axis, m0, m1, u - spec.post / 2, u + spec.post / 2,
                                       v1 + spec.embed, v1, v1 - spec.gap_t, spec.gap_t,
                                       a_lo, a_hi))
    return cut, fill


def subtract(mesh, parts):
    parts = list(parts)
    if not parts:
        return mesh
    return trimesh.boolean.difference([mesh] + parts, engine="manifold")


def add(mesh, parts):
    parts = list(parts)
    if not parts:
        return mesh
    return trimesh.boolean.union([mesh] + parts, engine="manifold")


# ---------- 校验与导出 ----------

def check(mesh):
    """返回 watertight / 闭合面 / 连通体数 / 体积(cm3) / 尺寸

    watertight（trimesh 口径）要求每条边正好 2 个面；但「连接件与膜完全共面」时
    manifold 会把共面区合并、三角化出重复面，产生 4 面边——面仍是闭合的，没有洞。
    所以另给 closed：只要没有 1 面边（边界边＝真洞），就算闭合。判缺陷看 closed。
    """
    import manifold3d as m3
    man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                              tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    bodies = len(man.decompose())
    F = np.asarray(mesh.faces, dtype=np.int64)
    key = np.sort(np.concatenate([F, np.roll(F, -1, axis=1)], axis=1).reshape(-1, 2), axis=1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    holes = int((cnt == 1).sum())
    nonman = int((cnt > 2).sum())
    return {"watertight": bool(mesh.is_watertight), "closed": holes == 0,
            "holes": holes, "nonmanifold": nonman, "bodies": bodies,
            "vol_cm3": mesh.volume / 1000.0,
            "bbox": [round(float(x), 1) for x in (mesh.bounds[1] - mesh.bounds[0])]}


def clean(mesh):
    """经 manifold3d 重绕 + 合并顶点，避免 STL float32 舍入产生非流形边"""
    import manifold3d as m3
    man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                              tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    o = man.to_mesh()
    m = trimesh.Trimesh(np.asarray(o.vert_properties, dtype=np.float64),
                        np.asarray(o.tri_verts, dtype=np.int64), process=True)
    m.merge_vertices(digits_vertex=6)
    return m


def _drop_inverted(mesh):
    """扔掉负体积的壳（里朝外的碎片）。

    布尔运算在极窄的缝、张开柱坡根贴到膜边缘的地方，偶尔会留下一个“里朝外”的封闭壳：
    面片合法、也是闭合的，但体积为负 —— 它表示的是材料里的一个气孔，不是零件。
    不处理的话分体数 +1（校验报“膜脱落”，其实是气孔），切片器还会把它当实体。
    只扔这种负体积壳；正体积的分体一概不动 —— 真脱落必须照旧报错。
    """
    man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                              tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    comps = man.decompose()
    if len(comps) < 2 or not any(c.volume() < 0.0 for c in comps):
        return mesh
    keep = [c for c in comps if c.volume() >= 0.0]
    if not keep:
        return mesh
    out = keep[0]
    for c in keep[1:]:
        out = out + c
    me = out.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(me.vert_properties)[:, :3].astype(float),
                           faces=np.asarray(me.tri_verts).astype(np.int64), process=False)


def _drop_tiny(mesh, min_vol=1.0):
    """剔掉布尔在口尖角处留下的碎屑，并报告「还有几块是分开的」。

    变宽口（新月/泪滴）的尖角上，膜的逐段偏移会削下 0.1~0.3 mm³ 的小薄片：
    面片合法、`closed=True`，但打不出来，还会让 `bodies` 虚高。阈值 1 mm³ —— 真脱落的
    连接件是 5~10 mm³ 量级，不会被吞掉；真分离时返回的 `n_left` > 1，调用方要照旧告警。
    """
    man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                              tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    comps = man.decompose()
    if len(comps) < 2:
        return mesh, 0, 1
    keep = [c for c in comps if c.volume() >= min_vol]
    dropped = len(comps) - len(keep)
    if dropped == 0:
        return mesh, 0, len(comps)
    if not keep:
        return mesh, dropped, len(comps)
    out = keep[0]
    for c in keep[1:]:
        out = out + c
    me = out.to_mesh()
    return (trimesh.Trimesh(vertices=np.asarray(me.vert_properties)[:, :3].astype(float),
                            faces=np.asarray(me.tri_verts).astype(np.int64), process=False),
            dropped, len(keep))


def clean_export(mesh, path):
    """导出后重新载入校验；返回 dict（含 reload_watertight）"""
    m = _drop_inverted(clean(mesh))
    m.export(path)
    back = trimesh.load(path)
    info = check(back)
    info["reload_watertight"] = bool(back.is_watertight)
    info["path"] = path
    return info


# ---------- 自检 demo ----------

def demo(out_dir):
    """200×300×3 面板，两扇 150×110 的窗，出一版带薄墙、一版通孔"""
    res = {}
    for tag, skin_on in (("薄墙", True), ("通孔", False)):
        panel = trimesh.creation.box(extents=[200, 3, 300])
        panel.apply_translation([100, 1.5, 150])
        cut, fill = [], []
        for v0, v1 in ((30, 140), (170, 280)):
            c, f = window("y", 0.0, 3.0, 25.0, 175.0, v0, v1, r=8.0)
            cut += c
            if skin_on:
                fill += f
        panel = subtract(panel, cut)
        panel = add(panel, fill)
        p = os.path.join(out_dir, f"thinwall_demo_{tag}.stl")
        res[tag] = clean_export(panel, p)
        print(tag, res[tag])
    return res


# ==========================================================================
# v2：任意闭合折线（含内环/岛）的「居中膜 + 连接件」
#
# 与 v1 的区别：
#   * 口不再限于圆角矩形，输入是若干闭合折线（外环 + 可选内环）
#   * 膜由 manifold3d.CrossSection.offset 常数内缩得到（Miter 保形）
#   * 缝宽两类：张开侧按 W_top/30° 反算，其余 = 3×层高
#   * 连接件三类：上缘张开柱 / 下缘直条柱 / 非上下边薄墙片（净跨 ≤ 3）
# ==========================================================================

import manifold3d as m3

_EPS = 1e-9


def _sarea(pts):
    n = len(pts)
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                     for i in range(n))


def _unit2(v):
    l = math.hypot(v[0], v[1])
    return (v[0] / l, v[1] / l) if l > _EPS else (0.0, 0.0)


def _inside_ring(pt, ring):
    """射线法判点是否在外环内（只用于尖角件的落点，环很小的开销）"""
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if xc > x:
                inside = not inside
    return inside


def _pdist_ring(pt, ring):
    """点到环（多边形边界）的最短距离"""
    best = float("inf")
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy
        u = 0.0 if ll < _EPS else max(0.0, min(1.0, ((pt[0] - x0) * dx + (pt[1] - y0) * dy) / ll))
        qx, qy = x0 + dx * u, y0 + dy * u
        best = min(best, math.hypot(pt[0] - qx, pt[1] - qy))
    return best


def _march_out(ring_wall, p, axis, s_from, step=0.02, s_max=6.0):
    """从 p 沿 axis 往外走，返回第一个“走出墙面轮廓（= 落到墙料里）”的 s。

    注意不能用“离膜边界 ≥ 缝宽”当判据：尖角是折线顶点，膜外一点的最近点就是那个顶点，
    距离按 Δ 直线算，会在缝里就提前停住（实测顶尖那颗柱只到 0.7、根本没碰到墙）。
    墙面轮廓就是使用者给的窗口外环，越过去才是墙。
    """
    s = s_from
    while s < s_max:
        q = (p[0] + axis[0] * s, p[1] + axis[1] * s)
        if not _inside_ring(q, ring_wall):
            return s
        s += step
    return s_max


def _norm_rings(rings):
    """外环统一逆时针、内环统一顺时针（材料恒在左）"""
    out = []
    for k, r in enumerate(rings):
        pts = [(float(p[0]), float(p[1])) for p in r]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        a = _sarea(pts)
        if (k == 0 and a < 0) or (k > 0 and a > 0):
            pts = pts[::-1]
        out.append(pts)
    return out


@dataclass
class WindowSpec:
    """异型镂空口的参数。layer / lw / skin 必须由使用者给，不设推导默认。"""
    layer: float = 0.2          # 层高（每次开工先问）
    lw: float = 0.4             # 线宽（每次开工先问）
    skin: float = 0.45          # 膜厚（使用者给）
    wall: float = 3.0           # 墙厚（张开封顶用）
    up: tuple = (0.0, 1.0)      # up_dir（口的平面内），决定哪一侧是“上缘”
    gap_side: float = None      # 非张开侧缝宽，默认 max(3*layer, gap_min)
    gap_min: float = None       # 缝宽下限（可打印最小槽宽），None = lw。只有默认公式会被它抬起：
                                # 层高 0.08 时 3×layer=0.24 比一条挤出线还窄，切片器必然用间隙填充
                                # 把缝并实（缝就是膜与墙之间唯一的那层空腔），所以下限 = 线宽。
                                # 显式给 gap_side 时不改值，只报风险（见 print_risk）。
    flare: bool = True          # 上缘是否张开
    flare_deg: float = 30.0
    flare_wmax: float = 6.0
    min_membrane: float = 2.0   # 整口「两侧缝之间最宽的那处」小于该值就**不生成**（返回空件、不报错）。
                                # 量法：逐段取局部净宽 W、减去两侧的缝，取最大值（欧氏距离）。
                                # 0 = 关闭。默认 2.0：0.08 层高下缝被 1.5×线宽托底到 0.6，
                                # 净宽 < 2 + 2×0.6 = 3.2 的口连一条 2mm 的膜都凑不出来，做了也是废片。
    post_w: float = 2.0         # 上缘张开柱宽（定死 2mm，沿墙切线方向）
    post_b: float = 2.0         # 下缘直条柱沿边宽（定死 2mm；竖直长按缝宽算出，见 window_poly）
    rib_w: float = None         # 非上下边薄墙片宽，默认 lw
    span_max: float = 3.0       # 允许的净跨上限
    miter_limit: float = 2.0    # 膜拐角 Miter 截断
    overshoot: float = 2.5      # 切口沿墙法向溢出
    embed: float = 0.1          # 连接件远端埋进墙的深度（保证融合；设 0 会让柱脚与墙面共面，实测拱顶裂成 11 片、异型 2 片）
    overlap: float = 0.0        # 连接件近端向膜内的重叠量；0 = 柱顶与膜下沿齐平（实测四形状仍单片、水密、导出水密）
    gap_floor: float = None     # 窄处可收窄到的下限，默认 max(lw, 0.4)
    start_w: float = None       # 判 link 的起膜边：柱的竖直总长。None = **算出来**：(缝 + embed)/cos，直线底边即 3×layer + 0.1
    start_gap: float = 3.0      # 判 link 的起膜边：相邻两柱之间留的空（不是中心距）
    link_posts: bool = True     # 判 link 的起膜边：段两端各一颗柱 + 中间按 start_gap 铺（False = 走老规则）
    rib_t: float = None         # 侧边薄墙片在墙厚方向的厚度；None = 与膜同厚(skin)，给值则居中取该厚度（如 =layer，实测 0.2）
    flare_tack: float = 0.0     # 张开柱贴墙端离墙的空档；0 = 埋进墙 embed，>0 = 末端停在离墙面该值处、不再咬墙（绝对值，mm）
    flare_tack_k: float = 0.0   # 同上，但按层高倍数给：tack = k × layer（常用 0.8，层高 0.2 时即 0.16）。层高会变的口径一律用这个，不要写死毫米数
    rib_snap: bool = False      # 侧边薄墙片沿 up 的宽度对齐层高网格：柱心压到层带中心 (k+0.5)×layer，使它在切片里只落一个层带（须 rib_w ≤ layer）
    conn_fuse: float = 0.0      # 连接件穿进膜的深度（绝对值，mm）
    conn_fuse_k: float = 0.0    # 同上，但按线宽倍数给：fuse = k × lw（常用 1/3，线宽 0.45 时即 0.15）。线宽会变就用这个
    oblique_deg: float = 45.0   # 上/下缘判定容差角(°)：外法向与 ±up 的夹角 ≤ 该值才算上/下缘，其余一律算侧边（横条）。
    corner_deg: float = 20.0    # 真拐角判定(°)：相邻两段的切线夹角超过该值才算拐角。
                                # 曲线（圆/样条/STL 边界离散出来的密折线）上每个顶点只转几度，
                                # 若都算拐角：凹弧每个顶点被判成"凹角"→ 角格留空规则把整条弧的柱删光；
                                # 凸弧每个顶点又是区段界 → 每颗柱都被"只留一颗"削掉。0 = 每个顶点都算拐角（老口径）
                                # 45 = 老口径（36° 斜边会被算成下缘 → 竖直柱，星形/多边形实测混着很难看）；30 = 斜边也走横条
    tip_deg: float = 0.0        # 尖角判定(°)：膜轮廓内角小于该值算尖角 —— 尖角两侧都不留端柱（2.0/2.55 宽的柱
                                # 在尖角上放不平，端面压在尖角时左右侧面会插进墙里）。0 = 关（老口径）。星形 36° 尖角给 60
    tip_mode: str = "strut"     # 尖角件形态：
                                #   bar  = 横条同款尺寸（断面 rib_w×膜厚、长度跨到墙+embed），轴线只摆 up/hz；
                                #          上尖给竖直件、臂尖与下尖给水平件（使用者定法：下尖横、上尖不横）
                                #   zone = 按尖角朝向分区：朝上/朝下给竖直柱（宽 post_b），朝水平给横条
                                #   strut= 一根窄舌（宽 = 缝宽）沿角平分线跨到墙
                                #   wedge= 三角楔把整个 V 填实（尖角处不留缝）
                                #   none = 不放
    tip_embed: float | None = None  # bar 模式末端咬墙量；None = 用 spec.embed；0 = 只贴墙面（不咬）
    tip_overlap: float = 0.25
    top_clear: float = 0.16         # 上缘张开柱顶端与上方墙面的间隙（mm），**写死 0.16**（使用者定版）：
                                     # 托住悬垂面但不与它熔死；层高变了它**不跟着变**；
                                     # None = 老口径 2*layer（历史行为）；0 = 柱端埋进墙 embed

    var_gap: bool = False           # 变宽口（新月/泪滴这类两头窄中间宽的口）逐段压缝：窄端缝被压到
                                    # 可打印下限以下时，该处膜与墙粘牢（切片器会填死，等效实心），
                                    # 换"窄端不放柱"，其余照常。False = 老口径（整口一个缝宽）
    max_wall: float = 5.0           # 墙厚上限(mm)：≥ 该值**不报错、直接不生成**（厚墙自身能打印，
                                    # 一处悬垂只在薄壁上加一套）。0/None = 关闭闸门

    @property
    def slot_min(self):
        """可打印最小槽宽，默认 1.5×线宽（lw=0.4 时 = 0.6）。

        缝是膜与墙之间唯一的那层空腔，一旦被填实，膜就和墙连成一块。
        缝宽是**水平**间隙，取决于挤出线宽而不是层高：层高从 0.2 掉到 0.08 时
        3×layer 只有 0.24，比一条线还窄，切片器必然拿一条挤出线把缝填死。
        实测能保住膜的最小值是 0.6（= 1.5×0.4 线宽），所以小层高时由这条下限接管。
        """
        if self.gap_min is not None:
            return self.gap_min
        return 1.5 * self.lw

    @property
    def g0(self):
        """非张开侧缝宽。显式值原样用（只报风险）；默认公式被 slot_min 抬起。"""
        if self.gap_side is not None:
            return self.gap_side
        return max(3.0 * self.layer, self.slot_min)

    @property
    def gap_up(self):
        """上缘（张开侧）缝宽：max(4×layer, g0 + 1×layer)。

        老口径下两者恒等 4×layer（g0 = 3×layer）。层高小到 3×layer 被下限抬起时，
        上缘仍要多留一层：张开柱的坡根要退进膜里，缝够长才轮得到它长宽。
        """
        return max(4.0 * self.layer, self.g0 + self.layer)

    @property
    def gfloor(self):
        """收缝的下限：窄口允许把缝收到这里，但不能再低。

        与 `slot_min` 分工：`slot_min`（1.5×线宽）管"默认缝宽不许选得比这更窄"，
        `gfloor`（≈线宽）管"被挤窄时最多能挤到哪"。两者混用会把窄口判死——1.2 的槽
        在 1.5×线宽下限下永远保不住膜，而切 0.4 缝 + 0.4 膜实际能打。
        """
        if self.gap_floor is not None:
            return self.gap_floor
        return max(self.lw, 0.4)

    @property
    def gap_fill(self):
        """切片器开始间隙填充的槽宽阈值（= 2×线宽）：缝窄于它，几何上有缝、打印后没有。"""
        return 2.0 * self.lw

    @property
    def tack(self):
        """张开柱末端离墙的实际空档：显式毫米值优先，否则按层高的倍数算。"""
        if self.flare_tack:
            return self.flare_tack
        return self.flare_tack_k * self.layer

    @property
    def fuse(self):
        """连接件穿进膜的实际深度：显式毫米值优先，否则按线宽的倍数算。

        上限压到 `skin/2`：再深就可能从膜的另一面穿出去。
        """
        v = self.conn_fuse if self.conn_fuse else self.conn_fuse_k * self.lw
        return min(v, 0.5 * self.skin)


def flare_values(spec):
    """张开侧（上缘）几何。

    缝宽是硬口径：gap_up = max(4×layer, g0+layer)，不由坡度反算。
    最宽处 W_top = min(flare_wmax, wall − lw)：柱顶到墙面的接触面 = W_top × post_w，
    每侧增长 grow = (W_top − skin)/2，张开长度 run = grow/tan(30°) 由坡度反推。
    缝放不下整段 run 时，斜坡**允许退进膜面内**（s_tip 可取负），也就是柱子加长、
    多出的那截压在薄墙里——否则顶宽吃不到 wall − lw。
    沿边方向不加宽：柱沿边恒 post_w（要靠柱数把净跨压下来，不是把柱摊宽）。
    """
    t = math.tan(math.radians(spec.flare_deg))
    w_top = min(spec.flare_wmax, spec.wall - spec.lw)
    grow = max(0.0, 0.5 * (w_top - spec.skin))
    run = grow / t if grow > _EPS else 0.0
    return {"w_top": w_top, "grow": grow, "run": run, "gap": spec.gap_up}


def _to_axis(m, axis):
    """把规范坐标系 (X=u, Y=v, Z=a) 转到 axis 指定的摆放"""
    if axis == "z":
        return m
    if axis == "y":       # [u, a, v]
        T = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    elif axis == "x":     # [a, u, v]
        T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    else:
        raise ValueError("axis 只能是 x/y/z")
    c = m.copy()
    c.vertices = c.vertices @ T.T
    if np.linalg.det(T) < 0:      # 换轴是镜像，需翻面修复朝向
        c.invert()
    return c


def _poly_solid(polys, a0, a1):
    """polygons → 单个实体：正面积=外环加，负面积=孔减"""
    pos, neg = [], []
    for p in polys:
        (pos if _sarea(p) > 0 else neg).append(_prism("z", a0, a1, p))
    if not pos:
        raise ValueError("截面为空（窄处已保不住一片膜）")
    solid = _union(pos)
    for h in neg:
        solid = trimesh.boolean.difference([solid, h], engine="manifold")
    return solid


def _min_residual(rings, spec, g_used, fv):
    """收窄之后，最窄处那片膜还剩多宽（两条相对边：净距 − 两缝）。

    `_shrink` 只保证"轮廓片数不变"，不等于"膜还够一条线宽"。实测 1.2 宽的槽在
    0.6 缝下会被收到 0.58，膜只剩 0.04 宽的一条，closed/bodies 全过，但打不出来。
    """
    up = _unit2(spec.up)
    c45 = math.cos(math.radians(spec.oblique_deg))
    segs = []
    for ri, r in enumerate(rings):
        flip = -1.0 if ri > 0 else 1.0
        n = len(r)
        for i in range(n):
            p = r[i]
            q = r[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]

            ln = math.hypot(dx, dy)
            if ln < _EPS:
                continue
            out = (dy / ln, -dx / ln)
            dot = flip * (out[0] * up[0] + out[1] * up[1])
            segs.append((p, q, out, fv["gap"] if (spec.flare and dot >= c45) else g_used))
    best = None
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            oi, oj = segs[i][2], segs[j][2]
            if oi[0] * oj[0] + oi[1] * oj[1] > -0.7:
                continue
            if min(_pt_seg(segs[i][0], segs[j][0], segs[j][1]),
                   _pt_seg(segs[i][1], segs[j][0], segs[j][1]),
                   _pt_seg(segs[j][0], segs[i][0], segs[i][1]),
                   _pt_seg(segs[j][1], segs[i][0], segs[i][1])) < 1e-9:
                continue
            d = _seg_dist(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
            w = d - segs[i][3] - segs[j][3]
            if best is None or w < best:
                best = w
    return best


def _shrink(cs, spec, g_target):
    """常数内缩；窄处自动收窄缝宽以保一片，收到底还不行就报错。

    返回 (section, g_used, splits)：splits 记录是否发生过收窄。
    """
    n_expected = cs.num_contour()
    g = g_target
    floor = min(spec.gfloor, g_target)
    splits = []
    while True:
        sec = cs.offset(-g, m3.JoinType.Miter, miter_limit=spec.miter_limit)
        if (not sec.is_empty()) and sec.num_contour() == n_expected:
            return sec, g, splits
        splits.append((round(g, 3), sec.num_contour()))
        if g <= floor + _EPS:
            raise ValueError(
                "窄处已收窄到 %.3f 仍保不住一整片膜（contour %d → %d），请放宽该处缝宽或改轮廓"
                % (g, n_expected, sec.num_contour()))
        g = max(floor, g - 0.02)


def _dedupe(pts):
    """去掉连续重复点与闭合点"""
    out = []
    for p in pts:
        if out and math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) < 1e-7:
            continue
        out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-7:
        out.pop()
    return out


def _frames(ring):
    """每段的 (t, inn)；inn = −out 指向口内（缝的反方向）。退化段给 None。"""
    n = len(ring)
    fr = []
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        d = (q[0] - p[0], q[1] - p[1])
        ln = math.hypot(*d)
        if ln < _EPS:
            fr.append(None)
            continue
        t = (d[0] / ln, d[1] / ln)
        fr.append((t, (-t[1], t[0])))
    return fr


def _pt_seg(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    wx, wy = p[0] - a[0], p[1] - a[1]
    ll = ax * ax + ay * ay
    t = 0.0 if ll < _EPS else max(0.0, min(1.0, (wx * ax + wy * ay) / ll))
    return math.hypot(wx - t * ax, wy - t * ay)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_dist(a, b, c, d):
    """两线段的最短距离（相交算 0）"""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if o1 * o2 < -1e-12 and o3 * o4 < -1e-12:
        return 0.0
    return min(_pt_seg(a, c, d), _pt_seg(b, c, d), _pt_seg(c, a, b), _pt_seg(d, a, b))


def _check_too_narrow(rings, spec, fv, g_used):
    """相对的两条边净宽不够时，那段膜会被 offset 整条抹掉，而 num_contour 不变
    → 静默漏膜（实测 h=1.2/1.4 全空、1.6 只剩极窄一条、≥2.5 才正常）。

    判据：两条相对边（外法向夹角 >135°）的净距 < 两侧缝宽之和 + 最小残留膜宽(lw)。
    只查“有一侧缝更宽”的对；两边同宽的窄处走 _shrink 的收窄路径，不在这里报错。
    """
    up = _unit2(spec.up)
    c45 = math.cos(math.radians(spec.oblique_deg))
    segs = []
    for ri, r in enumerate(rings):
        flip = -1.0 if ri > 0 else 1.0
        n = len(r)
        for i in range(n):
            p, q = r[i], r[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]
            ln = math.hypot(dx, dy)
            if ln < _EPS:
                continue
            out = (dy / ln, -dx / ln)
            dot = flip * (out[0] * up[0] + out[1] * up[1])
            segs.append((p, q, out, fv["gap"] if dot >= c45 else g_used))
    worst = None
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            oi, oj = segs[i][2], segs[j][2]
            if oi[0] * oj[0] + oi[1] * oj[1] > -0.7:      # 只查相对的两条
                continue
            # 相邻两条（共用一个顶点）不算“相对”：尖角处两边的外法向差
            # 180°−尖角，36° 的星尖就落到 dot=−0.81 被当成“相对的两条”，
            # 距离当然是 0 → 误报“离顶部过窄”。五角星实测踩到。
            if min(_pt_seg(segs[i][0], segs[j][0], segs[j][1]),
                   _pt_seg(segs[i][1], segs[j][0], segs[j][1]),
                   _pt_seg(segs[j][0], segs[i][0], segs[i][1]),
                   _pt_seg(segs[j][1], segs[i][0], segs[i][1])) < 1e-9:
                continue
            gi, gj = segs[i][3], segs[j][3]
            if max(gi, gj) <= g_used + _EPS:              # 两边同宽 → 交给 _shrink
                continue
            need = gi + gj + spec.lw
            d = _seg_dist(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
            if d < need - 1e-9 and (worst is None or d - need < worst[0]):
                worst = (d - need, d, gi, gj)
    if worst is not None:
        _, d, gi, gj = worst
        raise ValueError(
            "离顶部过窄：口内有两条相对的边只隔 %.3f，各留缝 %.3f / %.3f 后中间只剩 %.3f，"
            "小于最小残留膜宽 lw=%.3f。这段膜会被 offset 整条抹掉、片数却不变（静默漏膜）。"
            "把该处净宽放宽到 ≥ %.3f（= 两缝 + lw），或设 flare=False 让该段与其余段同宽。"
            % (d, gi, gj, d - gi - gj, spec.lw, gi + gj + spec.lw))


def _offset_ring(pts, shift, lim):
    """逐段带上位移量 shift=[((nx,ny), d)]，顶点取相邻两条移位后直线的交点；
    交点飞太远（> lim）或两段共线时退回平均位移。"""
    n = len(pts)
    out = []
    for i in range(n):
        an, ad = shift[(i - 1) % n]
        bn, bd = shift[i]
        pa = (pts[(i - 1) % n][0] + an[0] * ad, pts[(i - 1) % n][1] + an[1] * ad)
        qa = (pts[i][0] + an[0] * ad, pts[i][1] + an[1] * ad)
        pb = (pts[i][0] + bn[0] * bd, pts[i][1] + bn[1] * bd)
        qb = (pts[(i + 1) % n][0] + bn[0] * bd, pts[(i + 1) % n][1] + bn[1] * bd)
        da = (qa[0] - pa[0], qa[1] - pa[1])
        db = (qb[0] - pb[0], qb[1] - pb[1])
        den = da[0] * db[1] - da[1] * db[0]
        avg = (pts[i][0] + 0.5 * (an[0] * ad + bn[0] * bd),
               pts[i][1] + 0.5 * (an[1] * ad + bn[1] * bd))
        if abs(den) < 1e-12:              # 两段近乎共线：取平均位移
            out.append(avg)
            continue
        k = ((pb[0] - pa[0]) * db[1] - (pb[1] - pa[1]) * db[0]) / den
        ip = (pa[0] + da[0] * k, pa[1] + da[1] * k)
        if math.hypot(ip[0] - pts[i][0], ip[1] - pts[i][1]) > lim:   # miter 过长：截断
            out.append(avg)
        else:
            out.append(ip)
    return _dedupe(out)


def _hit_dist(rings_pts, p, d):
    """从 p 沿单位方向 d 打射线，到最近一条边的距离（None = 没打到）。

    用途有二：①逐段量局部净宽（变宽口压缝）；②连接件长度直接量到墙，
    曲线上比 `(缝+咬墙)/cos` 更准（直线边上两者完全相等）。
    """
    best = None
    for pts in rings_pts:
        n = len(pts)
        for j in range(n):
            a, b = pts[j], pts[(j + 1) % n]
            ex, ey = b[0] - a[0], b[1] - a[1]
            den = ex * d[1] - d[0] * ey
            if abs(den) < 1e-12:
                continue
            wx, wy = a[0] - p[0], a[1] - p[1]
            t = (ex * wy - wx * ey) / den
            u = (d[0] * wy - d[1] * wx) / den
            if t > 1e-6 and -1e-9 <= u <= 1.0 + 1e-9:
                if best is None or t < best:
                    best = t
    return best


def _edge_width(pts, i, inn):
    """第 i 段中点沿内法向到对边的距离 = 这条缝所在的局部净宽。"""
    p = (0.5 * (pts[i][0] + pts[(i + 1) % len(pts)][0]),
         0.5 * (pts[i][1] + pts[(i + 1) % len(pts)][1]))
    t = _hit_dist([pts], p, inn)
    return t


def _edge_gaps(rings, spec, fv):
    """逐段缝宽：先取该段该有的缝（上缘 4×layer、其余 g0），再按局部净宽压到"膜至少还剩 w_min"。

    返回 (gaps, uniform)。uniform=True 说明没有任何一段被压 → 走原来的常数缝宽路径，
    几何与老版本逐位一致；False 才是变宽口（新月、泪滴、收口的槽）。

    为什么必须逐段压：恒定缝宽在窄端会算出**负**膜宽（缝比口还宽），offset 自交、
    膜碎成几片；逐段压到 `(W − w_min)/2` 后，窄端的缝变小、膜保持一条线宽，代价是
    窄端的缝窄于可打印下限（会被切片器填死，等效于窄端实心）——这个代价必须报出来。
    """
    w_min = max(spec.lw, spec.skin)
    up = _unit2(spec.up)
    c45 = math.cos(math.radians(spec.oblique_deg))
    gaps, uniform = [], True
    for ri, r in enumerate(rings):
        pts = _dedupe([(float(p[0]), float(p[1])) for p in r])
        fr = _frames(pts)
        flip = -1.0 if ri > 0 else 1.0
        gi = []
        for i, f in enumerate(fr):
            if f is None:
                gi.append(max(spec.g0, 0.05))
                continue
            t, inn = f
            o = (t[1], -t[0])
            want = fv["gap"] if (spec.flare and flip * (o[0] * up[0] + o[1] * up[1]) >= c45) else spec.g0
            W = _edge_width(pts, i, inn)
            allowed = 0.5 * (W - w_min) if W is not None else want
            g = min(want, max(0.05, allowed))
            if g < want - 1e-9:
                uniform = False
            gi.append(g)
        gaps.append(gi)
    return gaps, uniform


def _clamp_frac(rings, gaps, slot_min):
    """缝被压到可打印下限以下的边，占全部周长的比例。

    用来区分两件事：新月/泪滴这种"只有窄端被压"（比例小，可打）与"整口都很窄"的长槽
    （比例大，打出来是一堵实心鳍，不是窗口 → 该拒绝）。
    """
    tot = hit = 0.0
    for ri, r in enumerate(rings):
        pts = _dedupe([(float(p[0]), float(p[1])) for p in r])
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ln = math.hypot(b[0] - a[0], b[1] - a[1])
            tot += ln
            if i < len(gaps[ri]) and gaps[ri][i] < slot_min - 1e-9:
                hit += ln
    return (hit / tot) if tot > 0 else 0.0


def _membrane_var(rings, gaps, spec):
    """逐段带缝宽偏移出膜（变宽口）：窄处缝自动变小，膜始终保一条线宽。"""
    polys = []
    for ri, r in enumerate(rings):
        pts = _dedupe([(float(p[0]), float(p[1])) for p in r])
        fr = _frames(pts)
        shift = []
        for f, g in zip(fr, gaps[ri]):
            if f is None:
                shift.append(((0.0, 0.0), 0.0))
            else:
                shift.append((f[1], g))
        lim = max(spec.miter_limit * max(gaps[ri]), 1e-6)
        polys.append(np.array(_offset_ring(pts, shift, lim), dtype=float))
    return m3.CrossSection(polys)


def _widen_up(rings, spec, fv, g_in, mem):
    """上缘缝比其余段宽（4×layer vs 3×layer）时的膜。

    做法：把上缘那几段的边线整体往口内挪 (gap_up − g_in)，其余段不动，顶点取相邻
    移位后直线的交点，然后**统一做一次 offset(−g_in)**。一次 offset、零布尔。

    早期做法是「先 offset(−g_in) 得到膜，再切掉一条 [g_in, gap_up] 的环带」，环带由逐段
    quad 并集构成、端头沿切线外伸，只在上缘是单一直段时才与膜的 miter 轮廓重合；
    上缘一旦由多段方向不同的线组成（拱顶、正弦波、斜切角），两者逐段错开，膜被割成
    锯齿甚至碎块（实测拱顶 1→4 片、波浪顶 1→11 片，随之布尔报 Not all meshes are volumes）。
    """
    delta = fv["gap"] - g_in
    if (not spec.flare) or delta <= _EPS:
        return mem
    up = _unit2(spec.up)
    c45 = math.cos(math.radians(spec.oblique_deg))
    lim = max(spec.miter_limit * delta, 1e-6)
    new = []
    for ri, r in enumerate(rings):
        pts = _dedupe([(float(p[0]), float(p[1])) for p in r])
        n = len(pts)
        fr = _frames(pts)
        flip = -1.0 if ri > 0 else 1.0        # 内环（岛）的“上缘”判定镜像
        shift = []
        for f in fr:
            if f is None:
                shift.append(((0.0, 0.0), 0.0))
                continue
            t, inn = f
            o = (t[1], -t[0])
            shift.append((inn, delta if flip * (o[0] * up[0] + o[1] * up[1]) >= c45 else 0.0))
        new.append(_offset_ring(pts, shift, lim))
    cs = m3.CrossSection([np.array(r, dtype=float) for r in new])
    return cs.offset(-g_in, m3.JoinType.Miter, miter_limit=spec.miter_limit)


def _sample(ring, cum, s):
    """按弧长 s 取点与该处“指向缝”的法向（离开材料的方向）"""
    n = len(ring)
    s = s % cum[-1]
    for i in range(n):
        if s <= cum[i + 1] + _EPS:
            p, q = ring[i], ring[(i + 1) % n]
            ln = cum[i + 1] - cum[i]
            u = 0.0 if ln < _EPS else (s - cum[i]) / ln
            pt = (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)
            t = (0.0, 0.0) if ln < _EPS else ((q[0] - p[0]) / ln, (q[1] - p[1]) / ln)
            return pt, (t[1], -t[0])
    p = ring[0]
    return p, (0.0, 0.0)


def _snap_side_s(ring, cum, s, spec, up, a0, a1, w, d_s, d_e):
    """把侧边薄墙片的柱心沿轮廓挪到层带中心，使它在切片里只占一层。

    侧片的宽度轴就是 up（打印层高方向），所以沿 up 的占用 = w。
    层带边界落在 layer 的整数倍上，w ≤ layer 时柱心取 (k+0.5)×layer 才整好压在一条层里。
    **只能沿膜轮廓（切向）挪**：法向挪会改变跨缝长度，挪多了就够不到墙。
    段两端那颗（离段端 w/2、端面与膜端面共面）不动，否则拐角会开出 0.1 的缝。
    """
    if w > spec.layer + 1e-9:
        return s
    if s - a0 <= d_s + 1e-9 or a1 - s <= d_e + 1e-9:
        return s
    pt, seam = _sample(ring, cum, s)
    hc = pt[0] * up[0] + pt[1] * up[1]
    band = math.floor(hc / spec.layer) * spec.layer
    tu = -seam[1] * up[0] + seam[0] * up[1]      # 切向(-seam[1], seam[0]) 在 up 上的投影
    if abs(tu) < 0.3:
        return s
    cand = [band + (0.5 + i) * spec.layer for i in (-1, 0, 1)]
    cand.sort(key=lambda c: abs(c - hc))         # 先试最近的层带，越界再试隔壁那条
    for c in cand:
        s2 = s + (c - hc) / tu
        if a0 + d_s - 1e-9 <= s2 <= a1 - d_e + 1e-9:
            return s2 % cum[-1]
    return s


def _stations(ring, spec, fv, g_used, m0, m1, warn):
    """沿一条轮廓按净跨 ≤ span_max 布点，返回 [(s, w, g, extra, flare, zone)]"""
    up = _unit2(spec.up)
    c45 = math.cos(math.radians(spec.oblique_deg))   # 上/下缘的容差角（默认 45°）
    flip = -1.0 if _sarea(ring) < 0 else 1.0      # 孔环（岛）：“上缘”判定镜像
    n = len(ring)
    cum = [0.0]
    zone = []
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        d = (q[0] - p[0], q[1] - p[1])
        ln = math.hypot(*d)
        cum.append(cum[-1] + ln)
        if ln < _EPS:
            zone.append(zone[-1] if zone else "side")
            continue
        out = (d[1] / ln, -d[0] / ln)
        dot = flip * (out[0] * up[0] + out[1] * up[1])
        zone.append("up" if dot >= c45 else ("down" if dot <= -c45 else "side"))
    # ── 曲线口：短区段清理 ──
    # 圆/样条/STL 边界离散出来的密折线，在 45° 容差带附近法向会在阈值两侧来回跳，
    # 切出一堆只有一两条边（长 0.1~0.5）的假区段；每个区段两头都要布端柱，结果柱连成
    # 一整排、缝全被堵死。比 min_run 短的区段并进较长的邻居，重建到稳定（最多 64 轮）。
    L_all = cum[-1]

    def _arc(i0, i1):
        dd = cum[i1 + 1] - cum[i0]
        return dd + L_all if dd < -1e-12 else dd

    min_run = max(1.0, 1.5 * g_used)
    for _guard in range(64):
        if len({z for z in zone}) <= 1:
            break
        st = 0
        for i in range(n):
            if zone[i - 1] != zone[i]:
                st = i
                break
        runs_c = []
        i = st
        for _ in range(n):
            j = i
            while True:
                j = (j + 1) % n
                if j == i or zone[j] != zone[i]:
                    break
            runs_c.append([i, (j - 1) % n, zone[i]])
            i = j
            if i == st:
                break
        short = [k for k, r in enumerate(runs_c) if _arc(r[0], r[1]) < min_run - 1e-9]
        if len(runs_c) <= 2 or not short:
            break
        k = min(short, key=lambda q: _arc(runs_c[q][0], runs_c[q][1]))
        prev, nxt = runs_c[k - 1], runs_c[(k + 1) % len(runs_c)]
        into = (prev[2] if _arc(prev[0], prev[1]) >= _arc(nxt[0], nxt[1]) else nxt[2])
        i, i1 = runs_c[k][0], runs_c[k][1]
        while True:
            zone[i] = into
            if i == i1:
                break
            i = (i + 1) % n

    # 闭合轮廓首尾若同区，s=0 会把一段劈成两半：段界去重会误删一颗柱——圆形口实测
    # 下缘正中因此留了 7.4 的缺口（正好在最该有支撑的地方）。所以把轮廓转到一个
    # “区段起点”上，让 s=0 落在真拐角；返回前再把 s 换算回原参数系。
    start_i = 0
    for i in range(n):
        if zone[i - 1] != zone[i]:
            start_i = i
            break
    s_off = cum[start_i]
    if start_i:
        ring = list(ring[start_i:]) + list(ring[:start_i])
        zone = zone[start_i:] + zone[:start_i]
        cum = ([s - s_off for s in cum[start_i:]]
               + [c - s_off + cum[-1] for c in cum[1:start_i + 1]])
    runs = []
    for i, z in enumerate(zone):
        if runs and runs[-1][2] == z:
            runs[-1][1] = cum[i + 1]
            runs[-1][4] = (i + 1) % n
        else:
            runs.append([cum[i], cum[i + 1], z, i, (i + 1) % n])
    # 每个顶点是不是「真拐角」：相邻两段切线夹角 > corner_deg。曲线上的离散顶点只转
    # 几度（R40 圆 0.5mm 弦长时约 0.7°），不能算拐角——否则凹弧每个顶点都被判成凹角、
    # 整条弧的柱被"角格留空"删光。
    corner_v = []
    for i in range(n):
        p0, p1, p2 = ring[i - 1], ring[i], ring[(i + 1) % n]
        d0 = (p1[0] - p0[0], p1[1] - p0[1])
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        l0 = math.hypot(*d0) or 1e-9
        l1 = math.hypot(*d1) or 1e-9
        cc = (d0[0] * d1[0] + d0[1] * d1[1]) / (l0 * l1)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, cc))))
        corner_v.append(bool(turn > spec.corner_deg))
    # 每个顶点是不是「凹角」：洞的边界在此内角 270°，材料只占一个象限。
    # 这种角上那颗柱没法只靠“贴”连住墙——角格只有 g×g，柱宽往往大于它，
    # 要连墙只能从侧面钻进墙角（工字形 (120,60) 实测钻 0.1），所以两侧都不放柱，
    # 角格留空，由两臂最近的那颗夹住（空口 0.6 会被切片器间隙填充吃掉）。
    _sgn = 1.0 if _sarea(ring) > 0 else -1.0
    reflex_v = []
    for i in range(n):
        p0, p1, p2 = ring[i - 1], ring[i], ring[(i + 1) % n]
        d0 = (p1[0] - p0[0], p1[1] - p0[1])
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        cr = d0[0] * d1[1] - d0[1] * d1[0]
        reflex_v.append(bool(corner_v[i] and cr * _sgn < -1e-12))

    # 尖角（内角 < tip_deg，如星形 36° 尖）：柱的端面一旦压在尖角上，柱的左右侧面会插进墙里
    # （2.0/2.55 宽的柱在 36° 尖角上根本放不平，实测切片里就是"张开柱的左侧贴着墙"）。
    # 规则：尖角两侧都不留端柱；指向下方的尖角再补一颗竖直下柱（膜尖端下方正对墙，补了才有支撑）。
    sharp_v, tip_dn_v = [], []
    for i in range(n):
        p0, p1, p2 = ring[i - 1], ring[i], ring[(i + 1) % n]
        v0 = (p0[0] - p1[0], p0[1] - p1[1])
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        l0 = math.hypot(*v0) or 1e-9
        l1 = math.hypot(*v1) or 1e-9
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (v0[0] * v1[0] + v0[1] * v1[1]) / (l0 * l1)))))
        sharp_v.append(bool(corner_v[i] and spec.tip_deg and ang < spec.tip_deg))
        mid = (0.5 * (p0[0] + p2[0]), 0.5 * (p0[1] + p2[1]))
        d = (p1[0] - mid[0], p1[1] - mid[1])
        ld = math.hypot(*d) or 1e-9
        tip_dn_v.append(bool(sharp_v[-1] and (d[0] * up[0] + d[1] * up[1]) / ld < -0.5))

    w_up = spec.post_w
    w_dn = spec.post_b
    w_sd = _rib_w(spec)
    # 上缘缝宽：张开时是 4×layer，不张开时与其余段一致（否则柱会多埋进墙 gf−g0）
    gf = fv["gap"] if spec.flare else g_used

    # ── 第一遍：每段的宽度、两条端柱的位置 ──
    # 布点是在**膜的轮廓环**上做的（环本身已比口线内缩 g），所以端柱外端面取在
    # 段端 `w/2` 处 = 端面与膜的端面共面。不往外多跨一点：一跨就会伸进口线那格
    # 拐角、把柱和墙连成一块（旧版 `w/2 − 0.05` 就是这个毛病）。
    plan = []                      # [a0, a1, zone, w, g, link, n_run, d_s, d_e, i0, i1]
    for a0, a1, z, i0, i1 in runs:
        L = a1 - a0
        w = {"up": w_up, "down": w_dn, "side": w_sd}[z]
        if L < 1e-6:
            continue
        g = gf if z == "up" else g_used
        # 判 link 的起膜边（朝下、落差 ≤ 2×lw）：改为「段两端各一颗柱 + 中间按 start_gap 铺」。
        # 柱的方向按**打印轴**定，不跟该处缝法向走：下缘柱一律竖直（⊥ 水平面）、
        # 侧边薄片一律水平（∥ 水平面）。曲线上法向在转，若跟法向就会成扇形——
        # 斜位置那根既不像柱也不像片（用户 圆口 实测指出两次）。
        # 下缘柱：宽定死 2.0（沿边），竖直长**按缝算**：(缝 3×layer + 咬墙 0.1)/cos，
        #   直线底边就是 0.7（露在外面 0.6 = 一段缝高），曲线上斜位置按 1/cos 放长；
        #   杆顶停在膜端面，**不伸进膜**。
        # 侧边薄片：水平，长度同样算出来，够到墙为止。上缘张开柱仍是“坡”跟着缝走。
        link = z == "down" and g_used <= spec.gap_fill + 1e-9
        if link:                      # 与 start_plan 同一条：边线倾角 < 45° 才算 link
            pt, seam = _sample(ring, cum, 0.5 * (a0 + a1))
            t = (-seam[1], seam[0])
            link = math.degrees(math.asin(min(1.0, abs(t[0] * up[0] + t[1] * up[1])))) < 45.0
        if link and spec.link_posts:
            w = spec.post_b                                  # 沿边宽（2.0，定死）
        n_run = _sample(ring, cum, 0.5 * (a0 + a1))[1]
        plan.append([a0, a1, z, w, g, bool(link and spec.link_posts), n_run, 0.0, 0.0, i0, i1])

    nseg = len(plan)
    for i in range(nseg):
        cur = plan[i]
        L = cur[1] - cur[0]
        d = min(max(0.02, 0.5 * cur[3]), 0.5 * L)
        cur[7], cur[8] = d, d

    # ── 段界（两个区相接的拐角）只留一颗柱 ──
    # 两侧各落一颗时，两颗都把拐角那小块缝挡住，拐角上就夹出一个**封闭空腔**
    # （凸台左上角实测 0.55×0.55，图上看着像"两边连成一块"、缝过不去）。
    # 规则：一个拐角只留宽的那一颗（同宽时 down > up > side），窄的那颗退场，
    # 拐角那片缝就顺着剩下那段敞开。
    _pri = {"down": 2, "up": 1, "side": 0}
    keep_first = [True] * nseg
    keep_last = [True] * nseg
    for i in range(nseg):
        if reflex_v[plan[i][9]] or sharp_v[plan[i][9]]:   # 凹角 / 尖角：两侧都不放柱
            keep_first[i] = False
        if reflex_v[plan[i][10]] or sharp_v[plan[i][10]]:
            keep_last[i] = False
    for i in range(nseg):
        j = (i + 1) % nseg
        if nseg == 1 and j == 0:
            break
        a, b = plan[i], plan[j]
        # 两颗柱的“近边”离各自段端都不到一个缝宽 → 一定夹住拐角
        if a[8] - 0.5 * a[3] <= g_used + 1e-9 and b[7] - 0.5 * b[3] <= g_used + 1e-9:
            ka = (a[3], _pri[a[2]])
            kb = (b[3], _pri[b[2]])
            if ka < kb:
                keep_last[i] = False
            else:
                keep_first[j] = False

    # ── 第二遍：布点 ──
    raw = []
    for (a0, a1, z, w, g, link, n_run, d_s, d_e, _i0, _i1), kf, kl in zip(plan, keep_first, keep_last):
        L = a1 - a0
        if link:
            # 先定位段两端的两颗（端面与膜端面对齐），再把剩余净长按 start_gap 均分出
            # 中间的柱子：柱心距 = 沿边宽 w + 实际间隙
            lm = L - d_s - d_e - w                     # 两端柱各自占掉 w 之后剩的净长
            if lm <= 1e-9:
                ss = [a0 + d_s, a1 - d_e]
            else:
                m = max(0, int(round((lm - spec.start_gap) / (w + spec.start_gap))))
                gap = (lm - m * w) / (m + 1.0)
                ss = [a0 + d_s]
                s = a0 + d_s + 0.5 * w + gap + 0.5 * w
                for _ in range(m):
                    ss.append(s)
                    s += w + gap
                ss.append(a1 - d_e)
            if not kf and len(ss) > 1:
                ss = ss[1:]
            if not kl and len(ss) > 1:
                ss = ss[:-1]
            for s in ss:
                raw.append((s % cum[-1], w, g, 0.0, False, z, "up"))
            continue
        k = max(1, int(math.ceil(L / (spec.span_max + w))))
        ss = []
        for i in range(k + 1):
            s = a0 + L * i / k
            if i == 0:
                s = a0 + d_s
            elif i == k:
                s = a1 - d_e
            ss.append(s)
        if not kf and len(ss) > 1:
            ss = ss[1:]
        if not kl and len(ss) > 1:
            ss = ss[:-1]
        for s in ss:
            if z == "side" and (spec.rib_snap or w <= spec.layer + 1e-9):
                # 侧片默认只有一个层高厚：柱心压到层带中心，切片里才真正落在同一层
                s = _snap_side_s(ring, cum, s, spec, up, a0, a1, w, d_s, d_e)
            raw.append((s % cum[-1], w, g, 0.0, z == "up", z, {"down": "up", "side": "side"}.get(z, "seam")))
    # 角格留空（尖角/凹角）必须按弧长再滤一遍：keep_first/keep_last 只管**区段**两端，
    # 同一区段内部拐过去的角（如星形两条斜边都是 side、中间夹着尖角/凹角）管不到，
    # 实测会有 0.2 宽的片正好落在角上、端面插进墙里。判据：件端面到角点的弧长距离 < tack。
    corners = [cum[i] for i in range(n) if reflex_v[i] or sharp_v[i]]
    if corners:
        keep = []
        for item in raw:
            s, w = item[0], item[1]
            hit = False
            for sc in corners:
                d = abs((s - sc) % cum[-1])
                if min(d, cum[-1] - d) < 0.5 * w + spec.tack + 1e-6:
                    hit = True
                    break
            if not hit:
                keep.append(item)
        raw = keep
    # 指向下方的尖角：正中补一颗竖直下柱（宽 post_b，竖直长按缝算出、末端埋进墙 embed）。
    # 上面刚把角上的件滤掉，这里补一颗正好把膜尖端压在墙上，又不让 2.0 宽的柱斜插进墙。
    for i in range(n):
        if tip_dn_v[i] and not spec.tip_deg:
            raw.append((cum[i] % cum[-1], spec.post_b, g_used, 0.0, False, "down", "up"))
    if not raw:
        return []
    # 补柱：角格留空会滤掉某颗柱，留下超过 span_max 的净跨（实测 blob 的 down 段到 4.77——
    # 膜首层在那一段是长桥，必下垂）。这里按弧长补到「每段净跨 ≤ span_max」；补的点若落在
    # 角格内就跳过，宁可让复核继续报「净跨超标」，也不把柱塞进墙角。
    _allc = cum[-1]
    for _p in range(4):
        if len(raw) < 2:
            break
        raw.sort(key=lambda t: t[0])
        _add = []
        for _i in range(len(raw)):
            _a = raw[_i]
            _b = raw[(_i + 1) % len(raw)]
            _gap = (_b[0] - _a[0]) % _allc
            if _gap <= 1e-9 or _a[5] != _b[5]:
                continue
            _z = _a[5]
            _clear = _gap - 0.5 * (_a[1] + _b[1])
            if _clear <= spec.span_max + 5e-3:
                continue
            _w = 0.5 * (_a[1] + _b[1])
            _n = max(1, int(math.ceil(_gap / (spec.span_max + _w))))
            _step = _gap / _n
            for _j in range(1, _n):
                _s = (_a[0] + _step * _j) % _allc
                _ok = True
                for _sc in corners:
                    _d = abs((_s - _sc) % _allc)
                    if min(_d, _allc - _d) < 0.5 * _w + spec.tack + 1e-6:
                        _ok = False
                        break
                if not _ok:
                    # 落点正在拐角上：拐角正中不放柱（两颗柱挤在角上会互相削），但这一段
                    # 的膜确实没人撑——把柱沿弧长挪到角格外侧的最近位置补上。
                    # 实测用户件「测试1.stl」删掉两个刀口楔后，凹角处 down 净跨 6.673，
                    # 全靠这一条压到 3 以内。
                    _near = None
                    for _sc in corners:
                        _d = (_s - _sc) % _allc
                        _dd = min(_d, _allc - _d)
                        if _dd >= 0.5 * _w + spec.tack + 1e-6:
                            continue
                        if _near is None or _dd < _near[0]:
                            _near = (_dd, _sc, _d)
                    if _near is not None:
                        _need = 0.5 * _w + spec.tack + 0.05
                        _sc = _near[1]
                        _s = (_sc + _need) % _allc if _near[2] <= _allc * 0.5 else (_sc - _need) % _allc
                        _ok = True
                        warn.append(("凹角补柱", round(_s, 2), round(_clear, 3)))
                if _ok:
                    _add.append((_s, _w, _a[2], 0.0, _z == "up", _z,
                                 {"down": "up", "side": "side"}.get(_z, "seam")))
        if not _add:
            break
        raw = raw + _add

    # 合并段界上的重复点（两侧各出一颗）：取较宽那一档，多出的并进膜里
    raw.sort(key=lambda t: t[0])
    out = []
    for item in raw:
        if out and abs(item[0] - out[-1][0]) <= max(1e-4, 0.02):
            prev = out[-1]
            gmax = max(prev[2], item[2])
            extra = max(prev[2], item[2]) - min(prev[2], item[2]) if prev[5] != item[5] else prev[3]
            out[-1] = (prev[0], max(prev[1], item[1]), gmax, extra,
                       prev[4] or item[4], prev[5], prev[6])
            warn.append(("段界取宽", round(prev[0], 2), round(gmax, 3), round(extra, 3)))
        else:
            out.append(item)

    # 首尾同点（闭合轮廓的 s=0 与 s=L）合并
    if len(out) > 1 and abs(out[-1][0] - cum[-1]) < max(1e-4, 0.02):
        f, l = out[0], out[-1]
        out[0] = (f[0], max(f[1], l[1]), max(f[2], l[2]),
                  max(f[3], l[3]), f[4] or l[4], f[5], f[6])
        out.pop()
    # s 换算回调用方的参数系（原 ring 的弧长起点）
    return [((s + s_off) % cum[-1], w, g, extra, flare, zone, ax)
            for (s, w, g, extra, flare, zone, ax) in out]


def _sweep_prof(prof, pt, n, t, w):
    """剖面沿 t 方向扫掠成体：柱沿边恒宽 w（沿边不加宽）。"""
    # 去掉连续重复点与闭合点，避免退化边
    clean = []
    for p in prof:
        if clean and abs(p[0] - clean[-1][0]) < 1e-7 and abs(p[1] - clean[-1][1]) < 1e-7:
            continue
        clean.append(p)
    if len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) < 1e-7 and abs(clean[0][1] - clean[-1][1]) < 1e-7:
        clean.pop()
    prof = clean
    npt = len(prof)
    hw = [0.5 * w] * npt
    V = []
    for sgn in (-1.0, 1.0):
        for (s, a), h in zip(prof, hw):
            k = sgn * h
            V.append([pt[0] + n[0] * s + t[0] * k, pt[1] + n[1] * s + t[1] * k, a])
    faces = [[0, i + 1, i] for i in range(1, npt - 1)]
    faces += [[npt, npt + i, npt + i + 1] for i in range(1, npt - 1)]
    for i in range(npt):
        j = (i + 1) % npt
        faces.append([i, j, npt + j])
        faces.append([i, npt + j, npt + i])
    m = trimesh.Trimesh(np.array(V, dtype=float), np.array(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def print_risk(spec, fv=None):
    """打印可行性清单：只做算术，不碰几何。

      * 缝窄于 2×线宽 → 切片器的间隙填充会把它并成实体（几何上有缝，打印后没有）
      * 起膜边（朝下、下方是空或水平面的那条膜边）是悬空起步，首层下垂 ≈1 层高
    """
    fv = fv or flare_values(spec)
    out = []
    fill_line = spec.gap_fill
    floor = spec.slot_min
    g0 = spec.g0
    for name, g in (("其余段", g0), ("上缘", spec.gap_up if spec.flare else g0)):
        if g < spec.lw - 1e-9:
            out.append("缝宽不可打印：%s缝 %.3f < 线宽 %.3f —— 连一条挤出线都放不下，切片器必然"
                       "把缝填实，膜会和墙连成一块（把 gap_side / gap_min 抬到 ≥ %.2f）"
                       % (name, g, spec.lw, floor))
        elif g < floor - 1e-9:
            out.append("缝宽窄于可打印下限：%s缝 %.3f < %.3f（1.5×线宽），切片器会把缝当薄壁"
                       "单独补线、大概率并成实体；要么抬 gap_min 到 %.2f，要么改成窄缝版"
                       "（明确接受膜与墙粘死）" % (name, g, floor, floor))
        elif g < fill_line - 1e-9:
            out.append("缝宽提示：%s缝 %.3f 落在 [下限 %.2f, 2×线宽 %.2f) 区间 —— 切片器的"
                       "间隙填充阈值是 2×线宽，缝窄于它就有可能被一条补线填死；实测 0.6 在"
                       "只沿法向的缝上不补（0.2 层高那批算例一直这么用）。**务必看一遍切片"
                       "预览**：若缝被填，抬 gap_min 到 %.2f。" % (name, g, floor, fill_line, fill_line))
    if spec.skin < spec.lw - 1e-9:
        out.append("膜厚提醒：skin %.3f < 线宽 %.3f，膜只够一条线，切片器按薄壁单独处理"
                   "（建议 skin ∈ [lw, 2×lw]）" % (spec.skin, spec.lw))
    out.append("起膜边提醒：朝下那条膜边只有 %.3f 悬空起步（= 缝宽），首层下垂 ≈ %.3f（1 层高），"
               "余量 %.3f；落差 ≤ 2×线宽(%.3f) 的那些段已改成「宽 %.1f 的竖直柱"
               "（竖直长按缝算出 = 缝 %.3f + 咬墙 %.3f，柱顶停在膜端面）+ 中间柱间留 %.1f」"
               "（缝仍留着），更大的落差须看 start_plan"
               % (g0, spec.layer, g0 - spec.layer, 2.0 * spec.lw,
                  spec.post_b, g0, spec.embed, spec.start_gap))
    _tc = _top_clear(spec)
    if _tc and _tc < spec.layer - 1e-9:
        out.append("顶隙太小：上缘张开柱顶端离墙面只有 %.3fmm（< 一层高 %.3f），支撑会和上方"
                   "悬垂面熔在一起、拆的时候带掉墙皮；建议 2 层（%.3f）"
                   % (_tc, spec.layer, 2.0 * spec.layer))
    return out


def start_plan(rings, spec, fv, g_used, bridge_max=None, z_distance=None):
    """给每条「起膜边」（外法向朝下、打印时膜最先出现的那条边）定动作。

    判据直接照抄切片器“要不要生成支撑”的那四条（Orca/Bambu/Prusa 同构）：
      1. 只有向下的面算悬垂，且这面与水平的夹角 ≥ 阈值（默认 45°）才生成支撑；
         边线越接近竖直（= 剖面上的一个尖点），越不需要撑——倒三角形就是这一类
      2. 两端都有锚点、跨度 ≤ max_bridge_length（默认 10mm；单线薄墙保守取 3）时
         切片器按“桥”处理，不生成支撑
      3. 悬垂面离下方表面 ≤ z_distance 时不生成支撑——贴太近；切片器自己取 0.2~0.3，
         这里取 `2×lw`（间隙填充距离）：缝本来就会被填满，等于贴脸，只能合缝
      4. 悬空周长小于阈值不撑：为一个小尖点单独加柱没有意义

    动作：
      none = 不加东西（尖点自带支撑 / 能桥接）
      link = 柱加密（判 link 的那些起膜边：段两端各一颗柱，中间按 `start_gap`=3.0 的
             柱间空均分铺柱；柱宽 `post_b`=2.0、竖直长由缝算出 `(缝+embed)/cos`
             （直线底边 0.7），长边一律沿打印轴 +v（⊥ 水平面），柱顶停在膜端面、不进膜）
             ——**已落地**，由 `_stations()` 直接生成；缝不动，柱子仍是“脚咬进下方材料、顶与膜端面齐平”
      post = 柱（柱距收到 bridge_max≈3，与 link 同一套件，只是更松）：只给判定与理由，不改几何

    返回 [(段号, 段长, 边线与水平夹角°, 落差, 动作, 理由)]
    """
    bridge_max = 3.0 if bridge_max is None else bridge_max
    z_distance = spec.gap_fill if z_distance is None else z_distance
    up = _unit2(spec.up)
    c45 = math.cos(math.radians(spec.oblique_deg))
    out = []
    for ri, r in enumerate(rings):
        flip = -1.0 if ri > 0 else 1.0
        n = len(r)
        for i in range(n):
            p, q = r[i], r[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]
            ln = math.hypot(dx, dy)
            if ln < _EPS:
                continue
            t = (dx / ln, dy / ln)
            o = (t[1], -t[0])
            if flip * (o[0] * up[0] + o[1] * up[1]) > -c45:
                continue                                   # 不是起膜边
            ang = math.degrees(math.asin(min(1.0, abs(t[0] * up[0] + t[1] * up[1])))
                               )                      # 0=水平边，90=指向上的边
            z = g_used
            if ang >= 45.0:
                act, why = "none", "边线已接近竖直（剖面是一个尖点），自带支撑"
            elif z <= z_distance:
                act, why = "link", "落差 %.3f ≤ z_distance %.3f：贴脸，柱间照样会碰" % (z, z_distance)
            elif ln <= bridge_max:
                act, why = "none", "段长 %.2f ≤ 桥接上限 %.2f，跨得过去" % (ln, bridge_max)
            else:
                act, why = "post", "水平悬垂段长 %.2f > 桥接上限 %.2f，须把柱距收到 %.2f" % (
                    ln, bridge_max, bridge_max)
            out.append((i, round(ln, 2), round(ang, 1), round(z, 3), act, why))
    return out


def _top_clear(spec):
    """上缘张开柱顶端与上方墙面的间隙（mm）：默认**写死 0.16**（使用者定版，0.08 层高下正好 2 层）。

    这个间隙是**沿墙厚平面内**的退让（柱端面到悬垂面下方的墙面），不是 Z 向的 z_distance：
    柱从膜端面出发跨过缝、长到离墙 top_clear 处停住，悬垂面首层下垂到这根柱顶就被挡住，
    下垂量被限制在 top_clear 以内，同时柱不与墙面熔死（拆/掰的时候不撕墙皮）。

    注意它**不随层高变**：`None` 才退回老口径 2*layer，`0` 回到"柱端埋进墙 embed"。
    """
    v = getattr(spec, "top_clear", 0.16)
    if v is None:
        return 2.0 * spec.layer
    return max(0.0, float(v))


def _rib_w(spec):
    """侧边片 / 尖角横条的沿边宽：默认 = **一个层高**（侧片的宽度轴就是打印层高方向，
    所以它是"一层厚"的桥，不跨多层、不在缝里砌出一堵墙）。

    `rib_w` 显式给值时原样用（老口径是 lw，即 5 层——那会在缝里留下一条 0.4 高的墙）；
    老代码直接取 `spec.rib_w`，默认 None 时 `0.5 * W` 会崩——tip_mode=bar/zone 一用就炸。
    """
    return spec.rib_w if spec.rib_w is not None else spec.layer


def _in_poly(P, ring):
    """偶奇规则的点在环内判定（P: (N,2) 数组，ring: (M,2) 折线）。"""
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), dtype=bool)
    M = len(ring)
    for i in range(M):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % M]
        cond = ((y1 > y) != (y2 > y))
        if not cond.any():
            continue
        t = (y - y1) / (y2 - y1)
        xs = x1 + t * (x2 - x1)
        inside ^= cond & (x < xs)
    return inside


def _max_membrane(rings, spec, fv=None):
    """整口「两侧之间最宽的那处」的膜宽 = 2×R − 2×缝，R 是口的最大内切圆半径。

    为什么不用逐边射线量：窄槽（100×1.2）的端边沿口长方向射线能打到 100mm 外的另一头，
    量出的是槽长不是厚度，会把「0.04 的膜」当成「98.8 的膜」放过去。内切半径不会：
    100×1.2 的槽 R=0.6 → 膜 1.2−1.2=0；150×230 的口 R=75 → 膜 150−1.2=148.8。

    网格分辨率 96×96（够判 0.1mm 量级），返回 None 表示口退化到取不出内点。
    """
    pts = np.asarray([(float(p[0]), float(p[1])) for r in rings for p in r], dtype=float)
    if len(pts) < 3:
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    if span <= 1e-6:
        return None
    n = 96
    gx = np.linspace(lo[0], hi[0], n)
    gy = np.linspace(lo[1], hi[1], n)
    GX, GY = np.meshgrid(gx, gy)
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)
    inside = np.zeros(len(P), dtype=bool)
    for r in rings:
        inside ^= _in_poly(P, np.asarray(r, dtype=float))
    if not inside.any():
        return None
    d = np.full(len(P), np.inf)
    for r in rings:
        R = np.asarray(r, dtype=float)
        M = len(R)
        for i in range(M):
            a = R[i]
            b = R[(i + 1) % M]
            ab = b - a
            L2 = float(ab[0] * ab[0] + ab[1] * ab[1])
            if L2 <= 1e-18:
                continue
            t = ((P[:, 0] - a[0]) * ab[0] + (P[:, 1] - a[1]) * ab[1]) / L2
            t = np.clip(t, 0.0, 1.0)
            dx = P[:, 0] - (a[0] + t * ab[0])
            dy = P[:, 1] - (a[1] + t * ab[1])
            d = np.minimum(d, np.hypot(dx, dy))
    Rr = float(d[inside].max())
    g = spec.g0
    return 2.0 * Rr - 2.0 * g


def _spans(st, total, spec=None):
    """按区统计布点：颗数、净跨、沿边占用长度。

    st 是 `_stations()` 的返回；净跨 = 相邻两颗柱**沿边方向**的空档
    （柱沿边等宽，膜端与墙端是同一个值，所以只有这一个量）。
    """
    out = {}
    pts = sorted([(s, w, zone) for (s, w, _g, _e, _f, zone, _ax) in st])
    if not pts:
        return out
    for s, w, z in pts:
        _z = out.setdefault(z, {"n": 0, "clear_max": 0.0, "cover_len": 0.0})
        _z["n"] += 1
        _z["cover_len"] += w
    for i, (s, w, z) in enumerate(pts):
        s2, w2, z2 = pts[(i + 1) % len(pts)]
        if z != z2:
            continue
        d = (s2 - s) % total
        if d <= 1e-9:
            continue
        out[z]["clear_max"] = max(out[z]["clear_max"], d - 0.5 * (w + w2))
    return out


def window_poly(rings, a_lo, a_hi, spec=None, axis="z"):
    """异型口：返回 (cut_parts, fill_parts)。

    rings : 口的平面闭合折线列表（第 0 条为外环，其余为内环/岛），坐标是 (u, v)
    a_lo/a_hi : 墙厚两端（沿 axis）
    """
    spec = spec or WindowSpec()
    _maxw = getattr(spec, "max_wall", 0.0) or 0.0
    if _maxw > 0 and float(spec.wall) >= _maxw - 1e-9:
        # 厚墙不需要薄墙支撑：**不报错、不生成**，返回空件。调用方看 fill 为空就跳过这个口。
        return [], [], {
            "skipped": "wall_too_thick", "wall": float(spec.wall), "max_wall": _maxw,
            "warn": [("墙厚超上限，不生成", float(spec.wall),
                      "墙厚 %.3f ≥ 上限 %.3f：厚墙自身就能打印，这里不加薄墙支撑（返回空件，不是错误）。"
                      % (float(spec.wall), _maxw))],
            "support": {}, "gap": None, "gap_up": None,
        }
    rings = _norm_rings(rings)
    fv = flare_values(spec)
    _min_mem = getattr(spec, "min_membrane", 0.0) or 0.0
    _mw_used = None
    if _min_mem > 0:
        _mw = _max_membrane(rings, spec, fv)
        _mw_used = _mw
        if _mw is not None and _mw < _min_mem - 1e-9:
            # 整口太窄：两侧缝之间最宽处也放不下一片 min_membrane 宽的膜 → 不生成（返回空件，不是错误）
            return [], [], {
                "skipped": "membrane_too_narrow", "membrane_max": _mw,
                "min_membrane": _min_mem,
                "warn": [("口太窄，不生成", round(_mw, 3),
                          "两侧缝之间最宽也只有 %.3f（< 下限 %.3f）：这里放不下一片能打的膜，"
                          "不加薄墙支撑（返回空件，不是错误）。把该处净宽放到 ≥ %.3f 再试。"
                          % (_mw, _min_mem, _min_mem + 2.0 * max(spec.lw, spec.slot_min)))],
                "support": {}, "gap": None, "gap_up": None,
            }
    warn = list(print_risk(spec, fv))

    open_cs = m3.CrossSection([list(r) for r in rings])
    cut_cs = open_cs
    open_pts = [_dedupe([(float(p[0]), float(p[1])) for p in r]) for r in rings]
    gaps_var, gap_uniform = (None, True)
    if spec.flare and spec.var_gap:
        gaps_var, gap_uniform = _edge_gaps(rings, spec, fv)
    if not gap_uniform:
        # 变宽口（新月/泪滴/收口的槽）：恒定缝宽在窄端会算出负膜宽 → 逐段压缝，
        # 膜永远保一条线宽，代价是窄端的缝窄于可打印下限（切片器会填死、等效窄端实心）。
        membrane_cs = _membrane_var(rings, gaps_var, spec)
        n_expect = len(rings)
        if membrane_cs.is_empty() or membrane_cs.num_contour() != n_expect:
            raise ValueError(
                "变宽口收缝后膜不成立（轮廓 %d → %s 片，应保持 %d 片）：该口有地方净宽窄于"
                " 膜最薄要求（lw/skin 取大者）：几何上放不下一整片膜。把该处口放宽、或减小 lw。"
                % (n_expect, 0 if membrane_cs.is_empty() else membrane_cs.num_contour(),
                   n_expect))
        g_used = min(min(gi) for gi in gaps_var)
        _frac = _clamp_frac(rings, gaps_var, spec.slot_min)
        if _frac > 0.5:
            raise ValueError(
                "这个口整体太窄：%.0f%% 的周长上缝都被压到 1.5×线宽(%.3f) 以下，最窄处只剩 "
                "%.3f 的膜、且缝会被切片器填死 —— 打出来是一堵 0.45 宽的实心鳍，不是窗口。"
                "把口整体放到 ≥ %.3f，或减小 lw / 设 flare=False 只留膜不张开。"
                % (100.0 * _frac, spec.slot_min, max(spec.lw, spec.skin),
                   spec.skin + 2 * spec.slot_min))
        if g_used < spec.slot_min - 1e-9:
            warn.append(("窄端缝被压到下限", round(g_used, 3),
                         "变宽口的窄端把缝从 %.2f 压到 %.3f（< 1.5×线宽 %.2f）：窄端会被切片器"
                         "间隙填充并死，等效该处膜与墙粘牢；膜本身仍保 %.2f 宽，打得出。"
                         "要窄端也开口，把窄端净宽放到 ≥ %.3f" % (
                             fv["gap"] if fv["gap"] > g_used else spec.g0, g_used,
                             spec.slot_min, max(spec.lw, spec.skin),
                             max(spec.lw, spec.skin) + 2 * spec.slot_min)))
    else:
        if spec.flare:
            # 先按名义缝宽把「相对两边夹得太死、整片膜会被 offset 抹掉」算清楚再动手收缝：
            # _shrink 收到底线也会抛错，但它的报错只说“保不住膜”，指不到是哪两条边、差多少。
            _check_too_narrow(rings, spec, fv, spec.g0)
        membrane_cs, g_used, splits = _shrink(open_cs, spec, spec.g0)
        if splits:
            warn.append(("窄处收窄", g_used, splits))
        if g_used < spec.slot_min - 1e-9:
            warn.append(("窄缝已收到下限", g_used,
                         "窄口把缝从 %.2f 收到 %.2f（< 1.5×线宽 %.2f）：膜保住了，但这些缝会被"
                         "切片器间隙填充并死，等于膜在该处与墙粘牢" % (spec.g0, g_used, spec.slot_min)))
        _w_min = _min_residual(rings, spec, g_used, fv)
        if _w_min is not None and _w_min < spec.lw - 1e-6:
            warn.append(("残留膜过窄", round(_w_min, 3),
                         "窄处把缝收到 %.2f 后，最窄那片膜只剩 %.3f 宽（< 线宽 %.3f）：几何闭合、"
                         "片数对，但切片器打不出比线宽更薄的一片，实际会断/塌。把该处净宽放到 "
                         "≥ %.3f，或设 gap_floor 更小让缝再收窄" % (g_used, _w_min, spec.lw,
                                                                  _w_min + 2 * g_used)))

        if spec.flare:
            _check_too_narrow(rings, spec, fv, g_used)
            membrane_cs = _widen_up(rings, spec, fv, g_used, membrane_cs)
            n_expect = len(rings)
            if membrane_cs.is_empty() or membrane_cs.num_contour() != n_expect:
                raise ValueError(
                    "加宽后膜不成立（轮廓 %d → %s 片，应保持 %d 片）：上缘缝 4×layer=%.3f 等于"
                    "在膜边再往里切 %.3f，该处局部净宽需 ≥ g_used(%.3f) + 4×layer(%.3f) + 最小残留膜宽"
                    "（建议 ≥ lw=%.3f）；否则设 flare=False（上缘与其他段同宽 3×layer），或放宽该处净宽。"
                    % (n_expect, 0 if membrane_cs.is_empty() else membrane_cs.num_contour(),
                       n_expect, fv["gap"], fv["gap"] - g_used, g_used, fv["gap"], spec.lw))

    m0 = a_lo + (a_hi - a_lo - spec.skin) / 2.0
    m1 = m0 + spec.skin

    cut = [_to_axis(_prism("z", a_lo - spec.overshoot, a_hi + spec.overshoot, r), axis)
           for r in rings]

    mem = _poly_solid(membrane_cs.to_polygons(), m0, m1)
    # 连接件与膜不许“对接”：`spec.fuse > 0`（= conn_fuse 或 conn_fuse_k×lw）时裁切用的膜整体内缩该值，连接件因此**穿进膜里**，
    # 和膜并出的是真体积重叠，界面不会留一对共面接触面。共面对接的坏处是切片器可能把
    # 0.2 的侧片当独立薄壁单独排一条线（实测现象：膜的喷嘴走线被挤向一边）。
    mem_clip = mem
    if spec.fuse > _EPS:
        shr = membrane_cs.offset(-spec.fuse, m3.JoinType.Miter,
                                 miter_limit=spec.miter_limit)
        if (not shr.is_empty()) and shr.num_contour() == membrane_cs.num_contour():
            mem_clip = _poly_solid(shr.to_polygons(), m0, m1)

    up = _unit2(spec.up)
    hz = (-up[1], up[0])
    bld = []                       # 规范坐标下的连接件：[mesh, 是否要按膜裁]
    n_ax, n_wedge, n_narrow = {}, 0, 0
    support, perim = {}, 0.0       # 布点统计（支撑够不够的直接量，见 _spans）
    for poly in membrane_cs.to_polygons():
        ring = [(float(p[0]), float(p[1])) for p in poly]
        n = len(ring)
        cum = [0.0]
        for i in range(n):
            q = ring[(i + 1) % n]
            cum.append(cum[-1] + math.hypot(q[0] - ring[i][0], q[1] - ring[i][1]))
        st = _stations(ring, spec, fv, g_used, m0, m1, warn)
        perim += cum[-1]
        for z, rec in _spans(st, cum[-1], spec).items():
            cur = support.setdefault(z, {"n": 0, "clear_max": 0.0, "cover_len": 0.0})
            cur["n"] += rec["n"]
            cur["cover_len"] += rec["cover_len"]
            cur["clear_max"] = max(cur["clear_max"], rec["clear_max"])
        for (s, w, g, extra, flare, zone, ax) in st:
            pt, seam = _sample(ring, cum, s)
            # 柱长直接量到墙：只有变宽口才需要（局部缝宽沿弧在变，常数缝宽算不准）；
            # 常数缝宽时仍用 (缝+咬墙)/cos —— 那套数被十几例实测钉过，不动它。
            d_hit = _hit_dist(open_pts, pt, seam) if spec.var_gap else None
            d_far = _hit_dist(open_pts, pt, (-seam[0], -seam[1])) if spec.var_gap else None
            if d_hit is not None and d_hit < spec.slot_min - 1e-9:
                # 变宽口的窄端：该处缝已压到可打印下限以下（切片器会填死、膜与墙粘牢），
                # 再放柱只会让 2.0 宽的柱伸出窄缝、被膜裁成悬空小块 → 这里干脆不放柱。
                n_narrow = n_narrow + 1
                continue
            if ax == "up":                # 下缘柱：竖直（⊥ 水平面），朝墙 = −up
                nrm, t = (-up[0], -up[1]), (hz[0], hz[1])
                cosv = abs(nrm[0] * seam[0] + nrm[1] * seam[1])
                if spec.start_w is not None:
                    s_end = spec.start_w
                elif d_hit is not None:
                    s_end = (d_hit + spec.embed) / max(0.35, cosv)
                else:
                    s_end = (g + spec.embed) / max(0.35, cosv)
            elif ax == "side":            # 侧边薄片：水平（∥ 水平面），长度够到墙为止
                nrm = hz if (hz[0] * seam[0] + hz[1] * seam[1]) >= 0.0 else (-hz[0], -hz[1])
                t = up
                cosv = abs(nrm[0] * seam[0] + nrm[1] * seam[1])
                s_end = ((d_hit + spec.embed) / max(0.35, cosv)) if d_hit is not None \
                    else (g + spec.embed) / max(0.35, cosv)
            else:                          # 上缘张开柱：跟着该处缝的坡走
                nrm, t = seam, (-seam[1], seam[0])
                # flare_tack>0：末端停在离墙面该值处（不咬墙，弱连接）；=0：照旧埋进墙 embed
                if spec.tack:
                    s_end = (d_hit - spec.tack) if d_hit is not None else (g - spec.tack)
                elif _top_clear(spec):
                    # 托的顶端停在离上方墙面 top_clear 处：托住但不与悬垂面熔死
                    _tc = _top_clear(spec)
                    s_end = ((d_hit - _tc) if d_hit is not None
                             else (g - _tc))
                else:
                    s_end = (d_hit + spec.embed) if d_hit is not None else (g + spec.embed)
            if ax == "side" and spec.rib_t:
                # 侧边薄墙片只取 rib_t 厚，居中放在墙厚中面上（默认与膜同厚）
                cc = 0.5 * (m0 + m1)
                r0 = cc - 0.5 * spec.rib_t
                r1 = cc + 0.5 * spec.rib_t
                prof = [(0.0, r0), (0.0, r1), (s_end, r1), (s_end, r0)]
            else:
                prof, s_tip = _profile(spec, fv, g, extra, flare, m0, m1, s_end=s_end)
            if (flare and d_hit is not None and d_far is not None
                    and -s_tip > d_far - d_hit + 1e-6):
                # 张开柱的坡根往膜里退 run−s_wide：变宽口的窄端膜只有一条线宽，
                # 坡根会从膜的另一侧穿出去、被裁成悬空小块（负体积）→ 这颗柱不放。
                n_narrow = n_narrow + 1
                continue
            bld.append([_sweep_prof(prof, pt, nrm, t, w), ax != "seam"])
            n_ax[ax] = n_ax.get(ax, 0) + 1

    if n_narrow:
        warn.append(("窄端不放柱", n_narrow,
                     "变宽口最窄的 %d 个布点上缝已压到 %.3f 以下：那里切片器会把缝填死、膜与墙"
                     "本来就连着，再放柱只会伸出窄缝变成悬空小块，所以不放。"
                     % (n_narrow, spec.slot_min)))

    # 轴对齐的连接件一律按膜裁一刀：柱/片只到膜端面为止，绝不再钻进膜里
    clip = [b[0] for b in bld if b[1]]
    parts = [mem] + [b[0] for b in bld if not b[1]]
    # ── 尖角件：走“端面在膜端点、侧面不跨过膜”的路子，两种形态（`tip_mode`）──
    #   strut（默认口径，使用者指定“用一个连接件把尖角处与墙连起来”）：一根窄舌。
    #     宽 = 缝宽 g_used（=3×层高），侧面与膜的两条臂在起点相切、往外只能落进更宽的缝里，
    #     所以侧面永不跨出膜；起点按 “半个宽 / tan(半角)” 埋进膜里（36° 尖埋 0.92，软化尖角顶点的薄边），
    #     末端照常规埋墙 embed。侧面在末段会与墙的 V 贴合约 1mm —— 尖角缝只有 0.6 厚，任何锚点都免不了。
    #   wedge：把整个 V 填实的三角楔（v4 旧口径，使用者否掉：“不这样子”）。
    if spec.tip_deg and spec.tip_deg > 0 and spec.tip_mode != "none":
        wall_ring = [(float(q[0]), float(q[1])) for q in rings[0]]   # 窗口外环 = 墙料边界
        for kpoly, poly in enumerate(membrane_cs.to_polygons()):
            if kpoly > 0:
                break                      # 岛上的尖角暂不处理（外环之外的墙料判据不适用）
            ring = [(float(p[0]), float(p[1])) for p in poly]
            nn = len(ring)
            for i in range(nn):
                p, pr, nx = ring[i], ring[i - 1], ring[(i + 1) % nn]
                v0 = (pr[0] - p[0], pr[1] - p[1])
                v1 = (nx[0] - p[0], nx[1] - p[1])
                l0, l1 = math.hypot(*v0), math.hypot(*v1)
                if l0 < _EPS or l1 < _EPS:
                    continue
                ang = math.degrees(math.acos(max(-1.0, min(1.0, (v0[0] * v1[0] + v0[1] * v1[1]) / (l0 * l1)))))
                if ang >= spec.tip_deg:
                    continue
                u1, u2 = (v0[0] / l0, v0[1] / l0), (v1[0] / l1, v1[1] / l1)
                half = math.radians(ang) / 2.0
                bx, by = u1[0] + u2[0], u1[1] + u2[1]
                lb = math.hypot(bx, by) or 1.0
                wo = (-bx / lb, -by / lb)               # 外向角平分线
                tt = (-wo[1], wo[0])
                if spec.tip_mode == "zone":
                    # 尖角件按尖角朝向分区，判法与边缘同源：朝上 → 上柱、朝下 → 下柱（都宽 post_b、竖直）、
                    # 其余 → 水平件（窄向 = rib_w 一层、长跨到墙），阈值取 45°
                    dn = wo[0] * up[0] + wo[1] * up[1]
                    c45 = math.cos(math.radians(spec.oblique_deg))
                    if dn >= c45:
                        ax_dir, W, zak = up, spec.post_b, "up"
                    elif -dn >= c45:
                        ax_dir, W, zak = (-up[0], -up[1]), spec.post_b, "down"
                    else:
                        W, zak = _rib_w(spec), "side"
                        ax_dir = hz if (wo[0] * hz[0] + wo[1] * hz[1]) >= 0.0 else (-hz[0], -hz[1])
                    lt = (-ax_dir[1], ax_dir[0])
                    s1 = _march_out(wall_ring, p, ax_dir, -0.25) + spec.embed
                    quad = [(p[0] + ax_dir[0] * s1 + lt[0] * (0.5 * W), p[1] + ax_dir[1] * s1 + lt[1] * (0.5 * W)),
                            (p[0] + ax_dir[0] * s1 - lt[0] * (0.5 * W), p[1] + ax_dir[1] * s1 - lt[1] * (0.5 * W)),
                            (p[0] - ax_dir[0] * 0.25 - lt[0] * (0.5 * W), p[1] - ax_dir[1] * 0.25 - lt[1] * (0.5 * W)),
                            (p[0] - ax_dir[0] * 0.25 + lt[0] * (0.5 * W), p[1] - ax_dir[1] * 0.25 + lt[1] * (0.5 * W))]
                    if _sarea(quad) < 0:
                        quad.reverse()
                    parts.append(_poly_solid([quad], m0, m1))
                    n_ax["tip_" + zak] = n_ax.get("tip_" + zak, 0) + 1
                elif spec.tip_mode == "bar":
                    # 尖角件照现有横条同一套尺寸口径：断面 = rib_w(一层，0.2) × 膜厚(线宽 0.45)，
                    # 长度 = 从膜端点跨过缝、末端咬进墙 embed（用窗口外环量，不能用“离膜 ≥ 缝宽”量）。
                    # 轴线只摆 up / hz 两个轴对齐方向：上尖给竖直件（“上尖不横”），臂尖与下尖给水平件（“下尖横”）
                    dn = wo[0] * up[0] + wo[1] * up[1]
                    if dn >= math.cos(math.radians(spec.oblique_deg)):
                        ax_dir, zak = up, "up"
                    else:
                        zak = "side"
                        ax_dir = hz if (wo[0] * hz[0] + wo[1] * hz[1]) >= 0.0 else (-hz[0], -hz[1])
                    W = _rib_w(spec)
                    lt = (-ax_dir[1], ax_dir[0])
                    e_end = spec.embed if spec.tip_embed is None else spec.tip_embed
                    s1 = _march_out(wall_ring, p, ax_dir, 0.0) + e_end
                    s0 = -spec.tip_overlap
                    quad = [(p[0] + ax_dir[0] * s0 + lt[0] * (0.5 * W), p[1] + ax_dir[1] * s0 + lt[1] * (0.5 * W)),
                            (p[0] + ax_dir[0] * s1 + lt[0] * (0.5 * W), p[1] + ax_dir[1] * s1 + lt[1] * (0.5 * W)),
                            (p[0] + ax_dir[0] * s1 - lt[0] * (0.5 * W), p[1] + ax_dir[1] * s1 - lt[1] * (0.5 * W)),
                            (p[0] + ax_dir[0] * s0 - lt[0] * (0.5 * W), p[1] + ax_dir[1] * s0 - lt[1] * (0.5 * W))]
                    if _sarea(quad) < 0:
                        quad.reverse()
                    parts.append(_poly_solid([quad], m0, m1))
                    n_ax["tip_" + zak] = n_ax.get("tip_" + zak, 0) + 1
                elif spec.tip_mode == "strut":
                    W = g_used
                    # 起点按“半个宽 / tan(半角)”埋进膜里再多退一个 fuse：正好相切时两角与膜边**零体积接触**，
                    # 导出重载会报非流形（实测 41 条），退一个 fuse 就让两角严格落在膜里
                    s0 = -(0.5 * W) / max(1e-6, math.tan(half)) - spec.fuse
                    s1 = (g_used + spec.embed) / max(1e-6, math.sin(half)) + (0.5 * W) / max(1e-6, math.tan(half))
                    # s1 再往后加 (半个宽/tan半角)：光按角平分线算的话，只有舌头正中那一点埋进墙，
                    # 两个角还留在缝里（焊成一个三角点）。加这一截，端面两个角才都到 g+embed、整条端面都焊进墙
                    quad = [(p[0] + wo[0] * s0 + tt[0] * (0.5 * W), p[1] + wo[1] * s0 + tt[1] * (0.5 * W)),
                            (p[0] + wo[0] * s1 + tt[0] * (0.5 * W), p[1] + wo[1] * s1 + tt[1] * (0.5 * W)),
                            (p[0] + wo[0] * s1 - tt[0] * (0.5 * W), p[1] + wo[1] * s1 - tt[1] * (0.5 * W)),
                            (p[0] + wo[0] * s0 - tt[0] * (0.5 * W), p[1] + wo[1] * s0 - tt[1] * (0.5 * W))]
                    if _sarea(quad) < 0:
                        quad.reverse()
                    parts.append(_poly_solid([quad], m0, m1))
                else:                                    # wedge（填实 V，旧口径）
                    apex = (p[0] + bx / lb * spec.fuse, p[1] + by / lb * spec.fuse)
                    L = (g_used + spec.embed) / max(1e-6, math.sin(half)) + spec.fuse
                    tri = [apex, (apex[0] - u1[0] * L, apex[1] - u1[1] * L),
                           (apex[0] - u2[0] * L, apex[1] - u2[1] * L)]
                    if _sarea(tri) < 0:
                        tri.reverse()
                    parts.append(_poly_solid([tri], m0, m1))
                n_wedge += 1
    if clip:
        conn = _union(clip)
        conn = trimesh.boolean.difference([conn, mem_clip], engine="manifold")
        parts.append(conn)
    # 连接件与薄墙**先并成一个整体**，再把它当一片去并到面板上：
    # 这样导出的 STL 里箱体/连接处/膜是同一张闭合面，切片按一个区域排线
    _conn = _to_axis(_union(parts), axis)
    _conn, _n_tiny, _n_left = _drop_tiny(_conn)
    if _n_tiny:
        warn.append(("膜边碎屑", _n_tiny, "尖角处 %.2f 以下的小薄片已剔除（打不出来）" % 1.0))
    if _n_left > 1:
        warn.append(("连接件未连到膜", _n_left, "膜与连接件没有并成一片，检查该处的缝宽/窄端"))
    fill = [_conn]

    # ── 支撑复核：把「布点够不够撑住上方墙」落成可核对的数 ──
    for z, rec in support.items():
        rec["cover"] = (rec["cover_len"] / perim) if perim > 0.0 else 0.0
        if z == "up":
            rec["w_top_x"] = spec.post_w            # 沿边宽（不加宽）
            rec["contact_mm2"] = spec.post_w * fv["w_top"]   # 单颗柱贴住上方墙的搭接面积
        if rec["clear_max"] > spec.span_max + 5e-3:
            warn.append(("净跨超标", z, round(rec["clear_max"], 3), spec.span_max))
    if "up" in support and support["up"]["n"] == 0:
        warn.append(("上缘无柱", "上方墙首层整段没有搭接面，会整片悬垂"))

    return cut, fill, {"gap_side": g_used, "flare": fv, "warn": warn,
                       "plan": start_plan(rings, spec, fv, g_used),
                       "membrane_max": _mw_used, "support": support,
                       "conn_axes": dict(n_ax), "n_conn": len(bld), "n_wedge": n_wedge}


def _profile(spec, fv, g, extra, flare, m0, m1, s_end=None):
    """(s, a) 剖面。s 从膜边起、朝墙为正；extra>0 表示向膜内再多伸一段（并进膜里）

    近端向膜内多伸 spec.overlap（默认 0 = 柱顶与膜下沿齐平，共面也能被布尔并成一片）；
    远端埋进墙 spec.embed（这一侧**不能**设 0：贴面会让柱脚脱开，实测拱顶裂成 11 片）。
    张开柱：
      * 末端一律 gap + embed 贴墙（tack 已废）
      * 坡度 30° 由 run 保证；缝放不下时斜坡**退进膜面内**（s_tip 为负），
        此时坡根即膜面内的点，整段楔形压在膜上，靠膜的体积保证融合

    返回 (剖面, s_tip)：s_tip 是坡根所在的 s。
    """
    e = extra + spec.overlap
    s_far = g + spec.embed if s_end is None else s_end
    if flare and fv["grow"] > _EPS:
        grow = fv["grow"]
        s_wide = s_far
        s_tip = s_wide - fv["run"]
        if s_tip > -e + 1e-6:
            return ([(-e, m0), (-e, m1), (s_tip, m1), (s_wide, m1 + grow),
                     (s_wide, m0 - grow), (s_tip, m0)], s_tip)
        # 斜坡退进膜里：坡根落在膜面上，不再保留近端平直段
        return ([(s_tip, m0), (s_tip, m1), (s_wide, m1 + grow), (s_wide, m0 - grow)], s_tip)
    return ([(-e, m0), (-e, m1), (s_far, m1), (s_far, m0)], -e)


# ---------- v2 自检 demo ----------

def _panel(u1, v1, thick):
    p = trimesh.creation.box(extents=[u1, thick, v1])
    p.apply_translation([u1 / 2.0, thick / 2.0, v1 / 2.0])
    return p


def _rings_rect(u0, u1, v0, v1):
    return [[(u0, v0), (u1, v0), (u1, v1), (u0, v1)]]


def _rings_circle(cu=100.0, cv=100.0, r=40.0, n=180):
    """圆形口（离散成 n 边折线）。闭合轮廓的 s=0 会落在某段中间，专测“转区段起点”那条"""
    return [[(cu + r * math.cos(2.0 * math.pi * k / n),
              cv + r * math.sin(2.0 * math.pi * k / n)) for k in range(n)]]


def _rings_i():
    """工字形：上下翼缘 40..160、腹板 80..120（12 条边，含 4 个凹角）"""
    return [[(40, 30), (160, 30), (160, 60), (120, 60), (120, 220), (160, 220),
             (160, 260), (40, 260), (40, 220), (80, 220), (80, 60), (40, 60)]]


def _rings_notch():
    return [[(25, 30), (60, 30), (60, 95), (52, 105), (60, 115), (60, 260),
             (25, 260)]]


def _rings_island():
    outer = [(25, 30), (175, 30), (175, 260), (25, 260)]
    island = [(85, 120), (115, 120), (115, 180), (85, 180)]
    return [outer, island]


def _rings_slot():
    """1.2mm 窄槽：必然触发 _shrink 收窄（3×layer=0.6 会把槽两侧缝吃掉），
    用来回归「窄口 + 张开必须显式报错」这条（加宽后膜不成立）。"""
    return [[(60, 20), (160, 20), (160, 21.2), (60, 21.2)]]


def _rings_arch():
    """半圆顶：上缘由 24 段不同方向的线组成（下缘是直线）。
    回归「上缘一弯，膜被割成碎块」这条：早期按逐段 quad 切环带时膜会裂成 4 片并布尔失败。"""
    pts = [(25.0, 30.0), (175.0, 30.0), (175.0, 150.0)]
    for i in range(25):
        th = math.pi * i / 24.0
        q = (100.0 + 75.0 * math.cos(th), 150.0 + 75.0 * math.sin(th))
        if math.hypot(q[0] - pts[-1][0], q[1] - pts[-1][1]) > 1e-6:
            pts.append(q)
    return [pts]


def _rings_tabs(h=25.0):
    """口内左右壁各伸出一个凸台（半岛），两尖端之间留一条竖缝。

    h = 凸台上表面到口顶的距离。h ≤ `g_used + gap_up`(算例 1.4) 时，那段膜会被 offset
    整条抹掉且片数不变 → 回归「离顶部过窄必须报错」这条。
    """
    tv = 260.0 - h
    return [[(25, 30), (175, 30), (175, 229), (132, 229), (132, tv), (175, tv),
             (175, 260), (25, 260), (25, tv), (68, tv), (68, 229), (25, 229)]]


def _rings_star(cu=100.0, cv=150.0, R=90.0, ratio=0.382, n=5, phase=90.0):
    """五角星：10 条边、5 个 36° 尖角、5 个 252° 凹角。

    尖角处相邻两边的外法向差 180°−36°=144°（dot=−0.81），会被 `_check_too_narrow`
    误当成“相对的两条边”，距离当然是 0 → 误报“离顶部过窄”。这条用例钉住那个修：
    相邻（共用顶点）的两条边不参与“相对”判定。
    """
    r = R * ratio
    pts = []
    for k in range(2 * n):
        ang = math.radians(phase + k * (360.0 / (2 * n)))
        rad = R if k % 2 == 0 else r
        pts.append((cu + rad * math.cos(ang), cv + rad * math.sin(ang)))
    return [pts]


def demo_poly(out_dir):
    """自检：正例（水密/单片/缝宽口径）+ 负例（窄口 + 上缘张开必须显式失败）"""
    res = {}
    spec = WindowSpec(layer=0.2, lw=0.4, skin=0.45, wall=3.0, up=(0.0, 1.0))
    spec_noflare = WindowSpec(layer=0.2, lw=0.4, skin=0.45, wall=3.0, up=(0.0, 1.0),
                              flare=False)
    cases = (("矩形", _rings_rect(25, 175, 30, 260), spec, "pass"),
             ("圆形", _rings_circle(), spec, "pass"),
             ("工字形", _rings_i(), spec, "pass"),             ("异型带凹口", _rings_notch(), spec, "pass"),
             ("带岛环", _rings_island(), spec, "pass"),
             ("拱顶_上缘多段", _rings_arch(), spec, "pass"),
             ("五角星_尖角凹角", _rings_star(), spec, "pass"),
             ("窄槽不张开", _rings_slot(), spec_noflare, "skip"),
             ("窄槽张开_口太窄跳过", _rings_slot(), spec, "skip"),
             ("离顶部过窄必失败", _rings_tabs(h=1.4), spec, "error"),
             ("矩形_起膜边_柱间空1.5", _rings_rect(25, 175, 30, 260),
              WindowSpec(layer=0.2, lw=0.4, skin=0.45, wall=3.0, up=(0.0, 1.0),
                         link_posts=True, start_gap=1.5), "pass"),
             ("矩形_起膜边柱照旧", _rings_rect(25, 175, 30, 260),
              WindowSpec(layer=0.2, lw=0.4, skin=0.45, wall=3.0, up=(0.0, 1.0),
                         link_posts=False), "pass"))
    for tag, rings, sp, expect in cases:
        panel = _panel(200, 300, 3.0)
        try:
            cut, fill, info = window_poly(rings, 0.0, 3.0, spec=sp, axis="y")
            if info.get("skipped"):
                if expect == "skip":
                    print(tag, "PASS（按预期跳过：%s）" % info["skipped"])
                    res[tag] = {"skipped": info["skipped"], "info": info}
                    continue
                raise AssertionError("[%s] 被跳过（%s），但用例期望生成" % (tag, info["skipped"]))
        except ValueError as e:
            if expect == "error":
                print(tag, "PASS（按预期拒绝）", e)
                res[tag] = {"expected_error": str(e)}
                continue
            raise
        if expect == "error":
            raise AssertionError("[%s] 本该拒绝，却生成了几何（膜被刮光却没报错）" % tag)
        if expect == "skip" and not info.get("skipped"):
            raise AssertionError("[%s] 本该按「口太窄」跳过，却生成了几何" % tag)
        panel = subtract(panel, cut)
        panel = add(panel, fill)
        p = os.path.join(out_dir, "thinwall_v2_%s.stl" % tag)
        res[tag] = clean_export(panel, p)
        res[tag]["info"] = info
        _assert(res[tag], tag, sp, expect == "shrink")
        print(tag, "PASS", res[tag]["bodies"], round(float(res[tag]["vol_cm3"]), 2),
              "闭合" if res[tag]["closed"] else "有洞",
              "非流形边 %d" % res[tag]["nonmanifold"] if res[tag]["nonmanifold"] else "",
              info["gap_side"])
    return res


def _assert(info, tag, spec, expect_shrink):
    """自检断言：闭合（无边界边）、单片、缝宽口径正确

    不用 watertight：完全共面的连接件会带来 4 面边（重复面），但面是闭合的、没有洞。
    """
    bad = []
    if not info["closed"]:
        bad.append("有边界边（真洞 %d 条）" % info["holes"])
    if info["bodies"] != 1:
        bad.append("膜脱落（bodies=%s）" % info["bodies"])
    if abs(info["info"]["flare"]["gap"] - spec.gap_up) > 1e-9:
        bad.append("上缘缝宽不是 max(4×layer, g0+1×layer)=%.3f" % spec.gap_up)
    g_side = info["info"]["gap_side"]
    if expect_shrink and g_side > spec.g0 + 1e-9:
        bad.append("窄处未触发收窄")
    if (not expect_shrink) and g_side < spec.g0 - 1e-9:
        bad.append("不该收窄却收窄了")
    if bad:
        raise AssertionError("[%s] " % tag + "；".join(bad))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="demo")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "demo-poly":
        demo_poly(a.out)
    else:
        demo(a.out)

