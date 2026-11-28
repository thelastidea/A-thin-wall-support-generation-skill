#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 thelastidea — SPDX-License-Identifier: AGPL-3.0-only
"""stl_window.py —— 从 STL 里抽出镂空孔边界 → 调 thinwall 引擎生成薄墙 → 布尔回网格 → 导出。

    python stl_window.py in.stl --out out.stl [--layer 0.08] [--lw 0.4] --skin 0.40

流程
    1. 载入 + 合并顶点（trimesh.load(process=True, force="mesh") / merge_vertices）
    2. 定面：三角面按单位法向聚成平面簇，取面积最大者（可用 --face-normal / --face-sel 指定）
    3. 定墙厚：找与所选面平行、朝反方向的最近平面簇，顶点投到法向轴上取直方图峰值 → a_lo/a_hi
    4. 抽孔：中厚平面 mesh_plane 截面 → 自己缝合二维闭合环 → 删外形环 → 判嵌套 → 得孔/岛
    5. 坐标变换：把面法向转到 +Y、打印向上转到 +Z；逐孔 window_poly(axis="y")
    6. 逐孔 subtract(cut) / add(fill)，最后旋回世界系 clean_export

坐标 / 厚度约定（本文件的权威口径）
    * 面法向 n_out 朝外（由三角面绕序给出；若 mesh.volume<0 先整网格翻转）。
    * 面平面 = { p·n_out = d_face }，面上一点 c = d_face * n_out。
    * 规范系（右手系，det(R)=+1，不镜像）：
          +X = u 轴（面内第三个方向，= Y × Z）
          +Y = 内法向 = -n_out       ← 「墙厚方向 a」
          +Z = 打印向上（--up 在面内的正交投影）
      R 的列 = [X_w, Y_w, Z_w]，即 规范坐标 → 世界坐标：p_world = R @ p_can + c
    * a_lo = 0（面本身）；a_hi = T（背面，沿内法向量出来的墙厚）。
      厚膜居中在 a_lo + (a_hi - a_lo - skin)/2，与引擎口径一致。
    * 截面 = 中厚平面（规范系 Y = T/2）；口的二维坐标 (u, v) = (规范 X, 规范 Z)，
      所以 spec.up = (0, 1) 就表示 +Z = 打印向上。
    * 引擎 axis="y" 的输出布局是 [u, a, v] = [X, Y, Z]，正好就是规范系；
      所有布尔都在规范系里做，最后整体乘 [R | c] 旋回世界系导出。

关于依赖
    本脚本只用 trimesh + numpy（不用 scipy/shapely/rtree）：
    mesh.section() 需要 scipy、mesh.contains() 需要 rtree，因此在没有这些包的
    环境里都不能用，所以二维环的缝合、点在多边形内、嵌套判定全部自带。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback

import numpy as np


# ============================================================ 依赖引导
def _bootstrap_site():
    """把 workspace/.libs/site 挂进 sys.path（调用方没设 PYTHONPATH 时兜底）"""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, ".libs", "site")
        if os.path.isdir(cand):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


try:                                            # noqa: E402
    import trimesh
except Exception:                               # pragma: no cover
    _bootstrap_site()
    import trimesh


def _import_engine():
    """import thinwall。stl_window.py 通常和 thinwall.py 同目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(3):
        if os.path.isfile(os.path.join(d, "thinwall.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            break
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    else:                                       # pragma: no cover
        raise SystemExit("找不到 thinwall.py（应与 stl_window.py 同目录）")
    import thinwall as tw
    return tw


# ============================================================ 常用数值
def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("零向量")
    return v / n


def _sarea(p):
    """二维多边形有符号面积（CCW > 0）"""
    p = np.asarray(p, float)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _hist_peak(vals, bin_w=None):
    """直方图峰值：返回 (峰值处均值, 该 bin 计数, 占比)。平面度判据用它。"""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0, 0.0
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo
    if span <= 1e-9:
        return float(vals.mean()), int(vals.size), 1.0
    w = float(bin_w) if bin_w else span / 60.0
    nb = max(1, min(int(math.floor(span / w)) + 1, 4000))
    w = span / nb
    ii = np.floor((vals - lo) / w).astype(np.int64)
    ii = np.clip(ii, 0, nb - 1)
    cnt = np.bincount(ii, minlength=nb)
    k = int(np.argmax(cnt))
    m = ii == k
    return float(vals[m].mean()), int(cnt[k]), float(cnt[k]) / float(vals.size)


def _vol(obj):
    """体积（mm³）。trimesh 的 `Trimesh.volume` 是属性，manifold3d 的 `Manifold.volume()` 是方法。

    两种对象混在一个文件里，统一走这里，避免"属性当方法调"抛 TypeError 后被 except 吞掉。
    """
    v = getattr(obj, "volume", None)
    if callable(v):
        v = v()
    return float(v)


def _heal(mesh):
    """修布尔输出里「位置重复的顶点」，返回 (mesh, 修掉的重复数)。

    manifold3d 的布尔结果常带坐标完全相同、索引却不同的顶点（本测试件上是 12 个）。
    它们在内存里还满足「每条边正好 2 个面」，但写进 STL 再读回来时 trimesh 会按位置
    把它们焊成一个 → 变成「一条边 4 个面」的非流形边，严格校验给 watertight=False
    （体积、闭合、体数都还是对的，切片器多半能忍，但没必要留这个尾巴）。

    做法：先按位置去重拿回一个「几何相同、拓扑略坏」的网格，再交给 manifold3d 重算一遍
    （它内部按容差重建拓扑），输出的就是干净流形。修不动就原样返回，绝不把件搞丢。
    """
    faces = np.asarray(mesh.faces)
    if len(mesh.vertices) == 0 or len(faces) == 0:
        return mesh, 0
    V32 = np.asarray(mesh.vertices, np.float32)
    uniq, inv = np.unique(V32, axis=0, return_inverse=True)
    ndup = len(V32) - len(uniq)
    t = trimesh.Trimesh(uniq.astype(np.float64), np.asarray(inv[faces], np.int64),
                        process=False)
    if ndup == 0 and t.is_watertight:
        return mesh, 0                      # 本来就好，一点不动
    try:
        import manifold3d as m3
        man = m3.Manifold(m3.Mesh(
            vert_properties=np.asarray(t.vertices, np.float32),
            tri_verts=np.asarray(t.faces, np.uint32)))
    except Exception as e:
        print("[警告] 顶点焊接失败：%s: %s" % (type(e).__name__, e))
        return mesh, 0
    if "NoError" not in str(man.status()):
        return mesh, ndup
    mm = man.to_mesh()
    h = trimesh.Trimesh(np.asarray(mm.vert_properties, np.float64),
                        np.asarray(mm.tri_verts, np.int64), process=False)
    if len(h.faces) == 0 or not h.is_watertight:
        return mesh, ndup
    return h, ndup


def _annular_cut(rings, a_lo, a_hi, spec):
    """把引擎给的「每环一块穿透刀」改成环形切刀：外环刀 − 各岛刀。

    `tw.window_poly` 对 rings 里每一条环都给一块实体切刀（岛那块也打穿墙）。对面板类
    算例（岛是「膜上还要开个洞」）没问题；但从 STL 里截出来的岛是**真材料**（例如口袋
    底上的凸台），照引擎的刀会把岛整个切掉、还在背墙上钻一个洞。所以在有岛时自己重算：
    只切「外环以内、岛以外」的那圈空气，岛的材料原样保留（膜上照旧留出岛的洞）。
    """
    tw = _import_engine()
    prism = []
    for r in rings:
        m = tw._prism("z", a_lo - spec.overshoot, a_hi + spec.overshoot, r)
        prism.append(tw._to_axis(m, "y"))
    if len(prism) == 1:
        return prism
    return [tw.subtract(prism[0], prism[1:])]


def _regularize(mesh, eps_mm3=0.5):
    """布尔/焊接残留的「零体积退化解片」清掉，再把剩下的连通体自并成一片。

    宽新月（两端收尖）这类口，manifold 布尔会留几个 |体积| < 0.5mm³ 的退化壳
    （实测 -0.35 / -0.23 mm³），`tw.check` 会把它算成额外 bodies，导出后切片器里
    就是几个大小为零的碎壳。做法：按连通体拆开 → 丢 |vol| ≤ eps → batch_boolean
    (Add) 自并（顺带把互相重叠的壳并干净）→ 重算拓扑。整体就没碎片时原样返回。
    返回 (mesh, info)，info 里 n_in/n_keep/drop 用于报告。
    """
    info = {"n_in": 1, "n_keep": 1, "drop": []}
    try:
        import manifold3d as m3
        man = m3.Manifold(m3.Mesh(
            vert_properties=np.asarray(mesh.vertices, np.float32),
            tri_verts=np.asarray(mesh.faces, np.uint32)))
    except Exception as e:
        print("[警告] 退化壳检查失败：%s: %s" % (type(e).__name__, e))
        return mesh, info
    if "NoError" not in str(man.status()):
        return mesh, info
    parts = list(man.decompose())
    info["n_in"] = len(parts)
    if len(parts) <= 1:
        return mesh, info
    keep, drop = [], []
    for p in parts:
        v = _vol(p)
        if abs(v) > eps_mm3:
            keep.append(p)
        else:
            drop.append(round(v / 1000.0, 6))
    info.update({"n_keep": len(keep), "drop": drop})
    if not keep:
        return mesh, info
    try:
        merged = m3.Manifold.batch_boolean(keep, m3.OpType.Add) if len(keep) > 1 else keep[0]
        mm = merged.to_mesh()
        h = trimesh.Trimesh(np.asarray(mm.vert_properties, np.float64),
                            np.asarray(mm.tri_verts, np.int64), process=False)
    except Exception as e:
        print("[警告] 重建拓扑失败：%s: %s" % (type(e).__name__, e))
        return mesh, info
    if len(h.faces) == 0 or not h.is_watertight:
        return mesh, info
    return h, info


def _weld(pts, tol):
    """把相距 <= tol 的二维点焊成同一个节点：返回 (uniq (M,2), idx (N,))"""
    pts = np.asarray(pts, float)
    tol = max(float(tol), 1e-9)
    tol2 = tol * tol
    uniq = []
    buckets = {}
    idx = np.empty(len(pts), np.int64)
    for i in range(len(pts)):
        p = pts[i]
        kx = int(math.floor(p[0] / tol))
        ky = int(math.floor(p[1] / tol))
        found = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((kx + dx, ky + dy), ()):
                    d = uniq[j] - p
                    if float(d[0] * d[0] + d[1] * d[1]) <= tol2:
                        found = j
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        if found < 0:
            found = len(uniq)
            uniq.append(p)
            buckets.setdefault((kx, ky), []).append(found)
        idx[i] = found
    return np.asarray(uniq, float).reshape(-1, 2), idx


def _loops_from_segments(seg2, tol):
    """二维线段 (N,2,2) → 闭合环列表。返回 (loops, n_open, n_pts)。

    平面截网格的每条线段来自一个三角面；相邻三角面共享边上的交点数值上略有差异，
    所以先按 tol 焊接点，再按「节点-边」相邻关系走环。非水密网格会留下开链，丢弃并计数。
    """
    seg2 = np.asarray(seg2, float).reshape(-1, 2, 2)
    if len(seg2) == 0:
        return [], 0, 0
    uniq, idx = _weld(seg2.reshape(-1, 2), tol)
    e = idx.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    if len(e) == 0:
        return [], 0, len(uniq)
    adj = {}
    for k in range(len(e)):
        a = int(e[k, 0])
        b = int(e[k, 1])
        adj.setdefault(a, []).append(k)
        adj.setdefault(b, []).append(k)
    used = np.zeros(len(e), bool)
    loops = []
    n_open = 0
    for k0 in range(len(e)):
        if used[k0]:
            continue
        used[k0] = True
        start = int(e[k0, 0])
        prev = start
        cur = int(e[k0, 1])
        chain = [start]
        closed = False
        while True:
            chain.append(cur)
            if cur == start:
                closed = True
                break
            cand = [kk for kk in adj.get(cur, ()) if not used[kk]]
            if not cand:
                break
            if len(cand) == 1:
                kk = cand[0]
            else:
                # 节点度数 > 2（三角面恰好穿过顶点）：选最「顺」的那条续边
                din = uniq[cur] - uniq[prev]
                kk = cand[0]
                bestdot = -2.0
                for kk2 in cand:
                    a2 = int(e[kk2, 0])
                    b2 = int(e[kk2, 1])
                    o = b2 if a2 == cur else a2
                    d2 = uniq[o] - uniq[cur]
                    nn = float(np.linalg.norm(din)) * float(np.linalg.norm(d2))
                    dot = float(np.dot(din, d2)) / (nn + 1e-12)
                    if dot > bestdot:
                        bestdot = dot
                        kk = kk2
            a2 = int(e[kk, 0])
            b2 = int(e[kk, 1])
            used[kk] = True
            prev = cur
            cur = b2 if a2 == cur else a2
            if len(chain) > 400000:             # 保险丝
                break
        if closed and len(chain) >= 5:
            loops.append(uniq[np.asarray(chain[:-1])])
        else:
            n_open += 1
    return loops, n_open, len(uniq)


def _simplify_loop(p, tol=1e-4):
    """去重合点、去共线点（截面的矩形外形会退化成 4 个角）"""
    p = np.asarray(p, float)
    out = []
    for q in p:
        if not out or float(np.hypot(q[0] - out[-1][0], q[1] - out[-1][1])) > tol:
            out.append(q)
    if len(out) > 1 and float(np.hypot(out[0][0] - out[-1][0],
                                      out[0][1] - out[-1][1])) <= tol:
        out.pop()
    n = len(out)
    if n < 3:
        return np.asarray(out, float).reshape(-1, 2)
    keep = []
    for i in range(n):
        a = out[(i - 1) % n]
        b = out[i]
        c = out[(i + 1) % n]
        ab = b - a
        bc = c - b
        lab = float(np.hypot(ab[0], ab[1]))
        if lab < 1e-12:
            continue
        h = abs(ab[0] * bc[1] - ab[1] * bc[0]) / lab      # c 到直线 ab 的垂距
        if h > tol:
            keep.append(b)
    if len(keep) < 3:
        keep = out
    return np.asarray(keep, float)


def _pt_in_poly(poly, p):
    """射线法（奇偶规则）点在多边形内判定"""
    poly = np.asarray(poly, float)
    x = float(p[0])
    y = float(p[1])
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    inside = False
    for i in range(len(poly)):
        if (y1[i] > y) != (y2[i] > y):
            xi = x1[i] + (y - y1[i]) * (x2[i] - x1[i]) / (y2[i] - y1[i])
            if x < xi:
                inside = not inside
    return inside


def _rep_point(poly):
    """嵌套判定的代表点：取**环上的顶点**。

    环与环互不相交，所以「环 i 上一点在环 j 内」⇔「环 i 被 j 包住」。
    **不能用质心**：环的质心可能落在它自己的孔里（比如「120×80 外形 + 中间的椭圆孔」，
    外形环的质心正好在椭圆孔内 → 会把外形和孔判成互相嵌套）。
    """
    return np.asarray(poly, float)[0]


# ============================================================ 定面：平面簇
def _plane_clusters(mesh, merge_deg=2.0, quant=0.01, quant_d=0.05, plane_tol=0.05):
    """把三角面按「法向 + 平面 offset」聚成平面簇。

    1) 法向按 quant(~0.6°) 量化、offset 按 quant_d 量化，一起分桶（向量化）
    2) 桶按角度 merge_deg + 平面偏移 plane_tol 贪心合并（面积加权）→ 平面簇
    3) 每个簇把顶点投到簇法向轴上取直方图峰值 → 平面 offset d、平面度

    只按法向聚类是不够的：盒子/口袋的「内壁」和「外壁」可能法向相同但在不同平面上，
    合并成一个簇后面积会虚高、offset 取到错的那个平面，窗口面/墙厚就全错了
    （实测 120×80×40 空心盒会挑到内后壁 d=-37 而不是前壁 d=0）。
    """
    fn = np.asarray(mesh.face_normals, float)
    fa = np.asarray(mesh.area_faces, float)
    ok = np.isfinite(fa) & (fa > 1e-12)
    if not ok.any():
        raise SystemExit("网格没有有效三角面")
    fn = fn[ok]
    fa = fa[ok]
    fidx = np.nonzero(ok)[0]

    tri_c = np.asarray(mesh.triangles_center, float)[fidx]
    fo = np.einsum("ij,ij->i", fn, tri_c)          # 每个面所在平面的 offset
    key = np.column_stack([np.round(fn / quant).astype(np.int64),
                           np.round(fo / quant_d).astype(np.int64)])
    order = np.lexsort((key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    grp = np.concatenate([[0], np.cumsum(~same)]).astype(np.int64)   # 每面的桶号
    nb = int(grp[-1]) + 1
    fsort = fn[order]
    asort = fa[order]
    osort = fo[order]
    bw = np.bincount(grp, weights=asort, minlength=nb)
    bn = np.column_stack([
        np.bincount(grp, weights=fsort[:, 0] * asort, minlength=nb),
        np.bincount(grp, weights=fsort[:, 1] * asort, minlength=nb),
        np.bincount(grp, weights=fsort[:, 2] * asort, minlength=nb),
    ])
    bo = np.bincount(grp, weights=osort * asort, minlength=nb) / np.maximum(bw, 1e-12)
    buckets = []
    for b in np.nonzero(bw > 1e-12)[0]:
        n = _unit(bn[b])
        buckets.append({"n": n, "area": float(bw[b]), "d": float(bo[b]),
                        "bids": [int(b)]})
    buckets.sort(key=lambda x: -x["area"])

    cos_merge = math.cos(math.radians(merge_deg))
    clusters = []
    for b in buckets:
        for cl in clusters:
            if (float(np.dot(b["n"], cl["n"])) > cos_merge
                    and abs(b["d"] - cl["d"]) <= plane_tol):
                tot = cl["area"] + b["area"]
                cl["n"] = _unit((cl["n"] * cl["area"] + b["n"] * b["area"]) / tot)
                cl["d"] = (cl["d"] * cl["area"] + b["d"] * b["area"]) / tot
                cl["area"] = tot
                cl["bids"].extend(b["bids"])
                break
        else:
            clusters.append({"n": b["n"].copy(), "area": float(b["area"]),
                             "d": float(b["d"]), "bids": list(b["bids"])})

    verts = np.asarray(mesh.vertices, float)
    for cl in clusters:
        mask = np.isin(grp, np.asarray(cl["bids"], np.int64))
        faces = fidx[order[mask]]
        cl["faces"] = faces
        cl["n_faces"] = int(len(faces))
        vids = np.unique(mesh.faces[faces].ravel())
        cl["vids"] = vids
        proj = verts[vids] @ cl["n"]
        d, cnt, frac = _hist_peak(proj)
        cl["d"] = d
        cl["planar"] = frac
        cl["peak_n"] = cnt
        cl["n_verts"] = int(len(vids))
        cl["point"] = cl["n"] * d
    clusters.sort(key=lambda c: -c["area"])
    return clusters


_AXES = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}


def _parse_axis(s):
    """'x'/'y'/'z'/'+y'/'-y'/'auto'/'-' → (方向单位向量 或 None, 是否指定了正负号)"""
    if s is None:
        return None, None
    t = str(s).strip().lower()
    if t in ("", "auto", "-"):
        return None, None
    sign = 1.0
    if t.startswith("+"):
        t = t[1:]
    elif t.startswith("-"):
        sign = -1.0
        t = t[1:]
    if t in _AXES:
        return sign * _AXES[t], (sign < 0 or str(s).strip().startswith("+"))
    raise SystemExit("--up / --face-normal 只认 x|y|z（可带 + / - 前缀），收到 %r" % s)


def _fmt_n(n):
    return "(%+.3f,%+.3f,%+.3f)" % (n[0], n[1], n[2])


def _print_faces(clusters, limit=60):
    print("%-4s %-26s %10s %6s %7s  %-26s %8s" %
          ("#", "法向(朝外)", "面积mm²", "面数", "平面度", "平面上一点", "offset"))
    for i, cl in enumerate(clusters[:limit]):
        print("%-4d %-26s %10.2f %6d %7.3f  %-26s %8.3f" %
              (i, _fmt_n(cl["n"]), cl["area"], cl["n_faces"], cl["planar"],
               "(%.3f,%.3f,%.3f)" % tuple(cl["point"]), cl["d"]))
    if len(clusters) > limit:
        print("...（只列出面积最大的 %d 个，共 %d 个；曲面/圆孔壁会各成一簇，属正常）"
              % (limit, len(clusters)))


def _pick_face(clusters, args):
    """选窗口所在面：--face-sel 优先，其次 --face-normal，最后面积最大的平面簇。

    --face-normal 指的是「窗口面朝外的方向」：
      * 带符号（-y）：只认外法向 ≈ 该方向的簇；一个都没有时，取法向反着的簇并把法向翻过来
        （允许用户把朝内方向写进参数）。
      * 不带符号（y）：±两个方向都算候选，再按面积挑。
    多个平行候选（凸台顶/沉台底/盒内壁与盒外壁法向相同）时，带符号方向取「沿该方向最靠外」
    的平面（offset 最大）、面积 tie-break —— 否则 120×80×40 空心盒会挑到内后壁(y=37)而不是
    前壁(y=0)。不带符号时朝向本来就二义，只能按面积挑。
    """
    if args.face_sel is not None:
        i = int(args.face_sel)
        if not (0 <= i < len(clusters)):
            raise SystemExit("--face-sel %d 越界（共 %d 个平面簇，先跑 --list-faces）"
                             % (i, len(clusters)))
        return clusters[i], "--face-sel %d" % i

    vec, signed = _parse_axis(args.face_normal)
    if vec is not None:
        cos_lim = math.cos(math.radians(3.0))
        flipped = False
        if signed is not None:
            cand = [cl for cl in clusters
                    if float(np.dot(cl["n"], vec)) >= cos_lim]
            if not cand:
                cand = [cl for cl in clusters
                        if float(np.dot(cl["n"], vec)) <= -cos_lim]
                flipped = bool(cand)
        else:
            cand = [cl for cl in clusters
                    if abs(float(np.dot(cl["n"], vec))) >= cos_lim]
        if not cand:
            raise SystemExit("没有外法向平行于 %s 的平面簇；用 --list-faces 看候选"
                             % args.face_normal)
        # 同向的簇可能把「法向相同、但不在同一平面」的面合并成一个（大而平面度低），
        # 用来当窗口面会错位。优先挑真正平坦的。
        for lim in (0.9, 0.5):
            flat = [cl for cl in cand if cl["planar"] >= lim]
            if flat:
                cand = flat
                break
        if signed is not None:
            # 沿给定方向的平面 offset（往朝外方向越远越大），再按面积 tie-break
            best = max(cand, key=lambda cl: (round(float(np.dot(cl["n"], vec)) * cl["d"], 3),
                                             cl["area"]))
        else:
            best = max(cand, key=lambda cl: cl["area"])
        how = "--face-normal %s" % args.face_normal
        if flipped:
            best["n"] = -np.asarray(best["n"], float)
            best["d"] = -float(best["d"])
            best["point"] = best["n"] * best["d"]
            how += "（该方向没有朝外的面，已翻转法向）"
        return best, how

    # auto：面积最大的平面簇（优先真正平坦的）
    for cl in clusters:
        if cl["planar"] >= 0.5:
            return cl, "auto(最大平面簇)"
    return clusters[0], "auto(最大平面簇,平面度偏低)"

    # auto：面积最大的平面簇（优先真正平坦的）
    for cl in clusters:
        if cl["planar"] >= 0.5:
            return cl, "auto(最大平面簇)"
    return clusters[0], "auto(最大平面簇,平面度偏低)"


# ============================================================ 定墙厚
def _detect_wall(mesh, clusters, face, args):
    """找与所选面平行、朝反方向、距离最近的平行面 → 墙厚 T。

    对每个候选平行簇：顶点投到所选面法向轴上取直方图峰值 → 背面 offset，
    T = d_face - d_back。面积太小/不平面的簇跳过（否则倒角小台阶会被当成背面）。
    """
    n = np.asarray(face["n"], float)
    d_face = float(face["d"])
    a_face = float(face["area"])
    cos_lim = math.cos(math.radians(3.0))
    verts = np.asarray(mesh.vertices, float)

    cands = []
    for cl in clusters:
        if cl is face:
            continue
        if float(np.dot(cl["n"], n)) > -cos_lim:
            continue                            # 法向不朝反方向 = 不是平行的背面
        proj = verts[cl["vids"]] @ n
        d, cnt, frac = _hist_peak(proj)
        T = d_face - d
        if T > 1e-6:
            cands.append({"T": T, "d": d, "area": cl["area"], "frac": frac,
                          "n_faces": cl["n_faces"]})
    cands.sort(key=lambda r: r["T"])

    passing = [r for r in cands
               if r["frac"] >= 0.5 and r["area"] >= max(0.01 * a_face, 4.0)]
    if passing:
        # 先按面积"够像背面"筛一道，再取**最近**的那张：
        #   · 空心壳（50×50 壳、壁 5）里，最近的平行面是腔体内壁（1600mm²，占 72%）→ T=5 正确；
        #   · 而沉台底/凸台面往往只有背面的 1%~2%，会被这道筛掉，免得把 T 量成台阶高度。
        # 老口径是"面积最大"，在空心壳里会一路选到远端的 50mm 大面，把 40×40 腔体整条打通。
        near = [r for r in passing if r["area"] >= 0.25 * max(a_face, 1e-9)]
        if near:
            r = min(near, key=lambda q: q["T"])
            return r["T"], ("平行面簇·最近够大(面积%.1fmm², 平面度%.2f)"
                            % (r["area"], r["frac"])), cands
        r = max(passing, key=lambda q: (q["area"], -q["T"]))
        return r["T"], ("平行面簇(面积%.1fmm², 平面度%.2f)" % (r["area"], r["frac"])), cands
    if cands:
        r = cands[0]
        return r["T"], ("最近平行面簇(面积%.1fmm², 平面度%.2f, 面积偏小/不平面)"
                        % (r["area"], r["frac"])), cands

    # 回退：不用平面簇，直接把全体顶点投到法向轴上找「下一个峰」
    proj = verts @ n
    sel = proj < d_face - 0.05
    if sel.any():
        d, cnt, frac = _hist_peak(proj[sel])
        T = d_face - d
        if T > 1e-6:
            return T, "背面顶点直方图峰(没找到平行面簇)", cands
    raise SystemExit("定不出墙厚：没找到与所选面平行的背面。请用 --wall 指定墙厚，"
                     "或用 --list-faces 换个面")


# ============================================================ 定坐标系
def _build_frame(n_out, up_hint, up_label):
    """规范系 R 的列 = [X_w, Y_w, Z_w]（规范→世界），det(R)=+1。

    +Y = 内法向 = -n_out；+Z = up 在面内的正交投影；+X = Y × Z。
    """
    Y = _unit(-np.asarray(n_out, float))
    v = np.asarray(up_hint, float)
    v = v - Y * float(np.dot(v, Y))
    L = float(np.linalg.norm(v))
    if L < math.sin(math.radians(5.0)):
        raise SystemExit(
            "--up %s 与面法向的夹角 < 5°（投影几乎为 0），定不出打印向上方向；"
            "换一个 --up（可用 --list-faces 看面法向再选）" % up_label)
    Z = v / L
    X = _unit(np.cross(Y, Z))
    R = np.column_stack([X, Y, Z])
    if abs(np.linalg.det(R) - 1.0) > 1e-9:
        raise SystemExit("内部错误：坐标系 det(R)=%r != +1" % np.linalg.det(R))
    return R, X, Y, Z


def _up_hint(args):
    """--up 解析：默认 +z；'-' 表示自动挑一个与面法向最不平行的世界轴。"""
    vec, signed = _parse_axis(args.up)
    if vec is not None:
        return vec, str(args.up)
    return None, "-"                                  # None = 稍后自动


# ============================================================ 截孔
def _section_regularize(loops, tol, tag=""):
    """截面 2D 规整：重叠/自交的孔环并成真轮廓，并把薄于 `tol` 的残留壁当不存在。

    为什么需要：从 STL 截出来的环可能自交（件是由几个互相重叠的空腔布尔出来的，
    重叠处留着两条几乎重合的壁）。这时环会来回交叉，引擎量到「两条相对的边只隔 0.0x」。

    做法：
      1) 外形环取面积最大的那个；其余按嵌套深度分「孔环」(深度 1) 与「岛环」(深度 ≥2)；
      2) 材料区域 = 外形 − NonZero(孔环) + NonZero(岛环)（NonZero 把自交环解释成它的并）；
      3) 若 `tol > 0`：对材料区域做 腐蚀 tol/2 + 膨胀 tol/2（形态学开运算）——
         薄于 `tol` 的材料整块消失，被它隔开的两个孔因此并成一个口。

    `tol` 默认 1.2×线宽：薄于这个值的壁切片器只会打成一条降宽"薄墙"，两侧孔壁的
    perimeter 还会把它吃掉，实际尺寸不可控。`tol=0` 只做环合并、不动材料。

    返回 `(loops2, info)`；`info` 里记着改前改后的环数/面积，便于把删掉的东西报出来。
    """
    import manifold3d as m3

    loops = [np.asarray(l, float) for l in loops]
    if len(loops) <= 1:
        return loops, None
    ar = [abs(float(_sarea(l))) for l in loops]
    i_out = int(np.argmax(ar))
    outer = loops[i_out]

    def _rep(l):
        try:
            return _rep_point(l)
        except Exception:                                    # noqa: BLE001
            return np.asarray(l, float).mean(axis=0)

    depth = []
    for i, l in enumerate(loops):
        if i == i_out:
            depth.append(0)
            continue
        p = _rep(l)
        d = 1 if _pt_in_poly(outer, p) else 0
        for j, lj in enumerate(loops):
            if j in (i, i_out):
                continue
            if _pt_in_poly(lj, p):
                d += 1
        depth.append(d)
    d1 = [loops[i] for i, dd in enumerate(depth) if dd == 1]
    d2 = [loops[i] for i, dd in enumerate(depth) if dd >= 2 and ar[i] < ar[i_out]]

    cs = m3.CrossSection([outer], m3.FillRule.NonZero)
    if d1:
        cs = cs - m3.CrossSection(d1, m3.FillRule.NonZero)
    if d2:
        cs = cs + m3.CrossSection(d2, m3.FillRule.NonZero)
    a_before = float(cs.area())
    n_before = int(cs.num_contour())
    if tol > 1e-9:
        t = 0.5 * float(tol)
        try:
            cs2 = cs.offset(-t, m3.JoinType.Round).offset(t, m3.JoinType.Round)
            if not cs2.is_empty():
                cs = cs2
        except Exception as e:                                # noqa: BLE001
            print("[警告] 薄壁开运算失败（跳过规整）：%s: %s" % (type(e).__name__, e))
    polys = [np.asarray(p, float) for p in cs.to_polygons()]
    info = {"rings_before": len(loops), "rings_after": len(polys),
            "depth1": len(d1), "depth2": len(d2),
            "area_before": a_before, "area_after": float(cs.area()),
            "contours_before": n_before, "contours_after": int(cs.num_contour())}
    info["dropped"] = info["area_before"] - info["area_after"]
    return polys, info


def _section_uv(mesh, origin, normal, X, Z, c, weld_tol):
    """中厚平面截网格 → 二维闭合环（规范系 (u,v) = (X, Z) 平面）"""
    # 注意：trimesh 5.x 的签名是 mesh_plane(mesh, plane_normal, plane_origin)，
    # 老版本是 (mesh, plane_origin, plane_normal)。一律用关键字，避开顺序差异。
    seg = trimesh.intersections.mesh_plane(mesh, plane_normal=normal,
                                           plane_origin=origin)
    if seg is None or len(seg) == 0:
        return [], 0, 0, 0
    seg = np.asarray(seg, float).reshape(-1, 2, 3)
    q = seg.reshape(-1, 3) - c
    uv = np.column_stack([q @ X, q @ Z]).reshape(-1, 2, 2)
    loops, n_open, n_pts = _loops_from_segments(uv, weld_tol)
    out = []
    for lp in loops:
        lp = _simplify_loop(lp)
        if len(lp) >= 3 and abs(_sarea(lp)) > 1e-6:
            out.append(lp)
    return out, n_open, n_pts, len(seg)



# ======================= 瓶颈局部剪除（split pinch） =======================

def _poly_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _seg_dist(a, b, c, d):
    """两条 2D 线段最短距离，返回 (dist, p_on_ab, p_on_cd)。"""
    ax, ay = float(a[0]), float(a[1]); bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1]); dx, dy = float(d[0]), float(d[1])
    ux, uy = bx - ax, by - ay
    vx, vy = dx - cx, dy - cy
    wx, wy = ax - cx, ay - cy
    a1 = ux * ux + uy * uy
    a2 = vx * vx + vy * vy
    b1 = ux * vx + uy * vy
    c1 = ux * wx + uy * wy
    c2 = vx * wx + vy * wy
    den = a1 * a2 - b1 * b1
    if a1 < 1e-18:
        s = 0.0
        t = (c2 / a2) if a2 > 0.0 else 0.0
    elif a2 < 1e-18:
        s = (c1 / a1) if a1 > 0.0 else 0.0
        t = 0.0
    elif den <= 1e-18:
        s = 0.0
        t = (c2 / a2) if a2 > 0.0 else 0.0
    else:
        s = (b1 * c2 - a2 * c1) / den
        t = (a1 * c2 - b1 * c1) / den
    s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    p = (ax + ux * s, ay + uy * s)
    q = (cx + vx * t, cy + vy * t)
    return (((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5), p, q


def _closest_pinch(poly, min_clear, min_arc=3.0, min_d=0.02, min_piece=4.0):
    """最窄的一对「相对」边：(dist, i, j, p, q) 或 None。

    判「相对」的三条同时成立才算窄颈，避免把普通凸边/三角化毛刺当瓶颈：
      1. 两条边基本平行（叉积模 < 0.3）；
      2. 两点连线基本垂直于这两条边（|cos| < 0.3）；
      3. 排除相邻边、自接触点（d < min_d）、沿环弧长挨得近的对（< min_arc）。
    """
    n = len(poly)
    if n < 8:
        return None
    arc = [0.0]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        arc.append(arc[-1] + ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    total = arc[-1]
    if total <= 0.0:
        return None
    best = None
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e1 = (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        l1 = (e1[0] ** 2 + e1[1] ** 2) ** 0.5
        if l1 < 1e-9:
            continue
        e1 = (e1[0] / l1, e1[1] / l1)
        for j in range(n):
            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                continue
            ds = abs(arc[i] - arc[j])
            ds = min(ds, total - ds)
            if ds < min_arc:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            e2 = (float(d[0]) - float(c[0]), float(d[1]) - float(c[1]))
            l2 = (e2[0] ** 2 + e2[1] ** 2) ** 0.5
            if l2 < 1e-9:
                continue
            e2 = (e2[0] / l2, e2[1] / l2)
            if abs(e1[0] * e2[1] - e1[1] * e2[0]) > 0.3:      # 不平行
                continue
            dd, p, q = _seg_dist(a, b, c, d)
            if dd >= min_clear or dd < min_d:
                continue
            w = (q[0] - p[0], q[1] - p[1])
            lw = (w[0] ** 2 + w[1] ** 2) ** 0.5
            if lw < 1e-9:
                continue
            w = (w[0] / lw, w[1] / lw)
            if abs(e1[0] * w[0] + e1[1] * w[1]) > 0.3:        # 连线不垂直于边
                continue
            if best is None or dd < best[0]:
                best = (dd, i, j, p, q)
    return best


def _cut_ring(poly, i, j, p, q):
    """第 i 边的 p 点与第 j 边的 q 点之间拉弦，把环切成两个环。"""
    n = len(poly)
    loop_a = [p] + [poly[(i + 1 + k) % n] for k in range((j - i) % n)] + [q]
    loop_b = [q] + [poly[(j + 1 + k) % n] for k in range((i - j) % n)] + [p]
    return loop_a, loop_b


def _min_dist_rings(poly_a, poly_b, sample=400):
    """两圈之间最近距离（边-边，点数多时按步长抽样控时间）。"""
    na, nb = len(poly_a), len(poly_b)
    if na < 2 or nb < 2:
        return float("inf")
    sa = max(1, na // sample)
    sb = max(1, nb // sample)
    best = float("inf")
    for i in range(0, na, sa):
        a, b = poly_a[i], poly_a[(i + 1) % na]
        for j in range(0, nb, sb):
            d, _, _ = _seg_dist(a, b, poly_b[j], poly_b[(j + 1) % nb])
            if d < best:
                best = d
                if best <= 1e-6:
                    return best
    return best


def _drop_pinched_islands(rings, need):
    """把与孔轮廓（或彼此）净距 < need 的岛当「楔尖」删掉。

    这种岛和孔边之间只剩一条比"两缝+膜"还窄的槽，膜根本过不去；留着它整孔必被拒。
    返回 (新 rings, 丢掉的岛列表, 最小净距)。
    """
    if len(rings) < 2:
        return rings, [], None
    outer = rings[0]
    kept = [outer]
    dropped = []
    mind = None
    for isl in rings[1:]:
        d = _min_dist_rings(outer, isl)
        for other in kept[1:]:
            d = min(d, _min_dist_rings(other, isl))
        if mind is None or d < mind:
            mind = d
        if d < need - 1e-9:
            dropped.append(isl)
        else:
            kept.append(isl)
    return kept, dropped, mind


def _split_pinches(rings, min_clear, max_pieces=8, max_iter=32):
    """把孔环在最窄瓶颈处逐个切开。

    返回 (子环组列表, 切点记录, 最窄值, 丢掉的岛数)；rings[0] 是孔轮廓，
    rings[1:] 是岛，按代表点归到包含它的子环，落在瓶颈上的岛被丢掉。
    """
    rings, dropped_isl, mind = _drop_pinched_islands(rings, min_clear)
    if dropped_isl:
        print("[剪除] 删掉 %d 个「楔尖」岛（与孔边最近 %.3fmm < %.3fmm，槽里塞不进膜）"
              % (len(dropped_isl), (mind if mind is not None else 0.0), min_clear))
    outer = [(float(x), float(y)) for x, y in rings[0]]
    if _poly_area(outer) < 0.0:
        outer = outer[::-1]                     # 统一 CCW
    pieces = [outer]
    cuts = []
    narrowest = None
    for _ in range(max_iter):
        if len(pieces) >= max_pieces:
            break
        hit_piece = None
        for pi in range(len(pieces)):
            hit = _closest_pinch(pieces[pi], min_clear)
            if hit is not None:
                hit_piece = (pi, hit)
                break
        if hit_piece is None:
            break
        pi, (d, i, j, p, q) = hit_piece
        a, b = _cut_ring(pieces[pi], i, j, p, q)
        if len(a) < 4 or len(b) < 4:
            break
        if min(abs(_poly_area(a)), abs(_poly_area(b))) < 4.0:
            break                       # 切出来是小碎块，说明这不是"两个大区之间的窄颈"

        narrowest = d if narrowest is None else min(narrowest, d)
        cuts.append((p, q, d))
        pieces[pi:pi + 1] = [a, b]

    isl = list(rings[1:]) if len(rings) > 1 else []
    out = []
    used = 0
    for pc in pieces:
        mine = [r for r in isl if len(r) >= 3
                and _pt_in_poly(pc, _rep_point(r))]
        used += len(mine)
        out.append([pc] + mine)
    return out, cuts, narrowest, max(0, len(isl) - used)


def _group_holes(loops):
    """按嵌套深度把环分成：外形环(深度 0) / 孔(深度 1) / 岛(深度 2) …

    返回 (holes, outer, deep)。holes 每项 = {"rings": [外环CCW, 岛CW...], "area": ...}
    """
    if not loops:
        return [], [], []
    rep = [_rep_point(l) for l in loops]
    depth = []
    for i in range(len(loops)):
        d = 0
        for j in range(len(loops)):
            if i == j:
                continue
            if _pt_in_poly(loops[j], rep[i]):
                d += 1
        depth.append(d)

    outer = [i for i, d in enumerate(depth) if d == 0]
    deep = [i for i, d in enumerate(depth) if d >= 3]
    holes = []
    for i, d in enumerate(depth):
        if d != 1:
            continue
        rings = [loops[i]]
        for j, dj in enumerate(depth):
            if dj == 2 and _pt_in_poly(loops[i], rep[j]):
                rings.append(loops[j])
        # 环朝向：外环 CCW(面积>0)，岛 CW(面积<0)。引擎自己也会归一化，这里显式做一遍便于报告。
        fixed = []
        r0 = np.asarray(rings[0], float)
        if _sarea(r0) < 0:
            r0 = r0[::-1]
        fixed.append(r0)
        for r in rings[1:]:
            r = np.asarray(r, float)
            if _sarea(r) > 0:
                r = r[::-1]
            fixed.append(r)
        holes.append({"rings": fixed, "area": abs(_sarea(loops[i])), "depth": depth[i]})
    return holes, outer, deep


def _edge_frames(ring):
    """环(CCW)每条边的单位切向、外法向、边长；返回 (P, t, nout, L)"""
    P = np.asarray(ring, float)
    if _sarea(P) < 0:
        P = P[::-1]
    d = np.roll(P, -1, axis=0) - P
    L = np.hypot(d[:, 0], d[:, 1])
    Ls = np.where(L < 1e-12, 1e-12, L)
    t = d / Ls[:, None]
    nout = np.stack([t[:, 1], -t[:, 0]], axis=1)      # CCW 环的外法向
    return P, t, nout, L


def _pt_in_ring(pt, poly):
    """点是否在多边形内（环绕数法）。

    用环绕数而不是射线交叉：射线法在「点正好落在与某条边共线的水平线附近、或与顶点
    同高」时会数错（尖角密集的轮廓尤其容易），实测五角星孔环就踩到过——同一个点射线法
    判「在内」、环绕数判「在外」，导致探针把有料的那一侧当成空腔。
    """
    x, y = float(pt[0]), float(pt[1])
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    a = np.arctan2(q[:, 1] - y, q[:, 0] - x) - np.arctan2(p[:, 1] - y, p[:, 0] - x)
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return abs(float(a.sum())) > math.pi


def _in_material(pt, polys):
    """奇偶规则：外形环 + 内孔环一起判，落在实心里返回 True。"""
    inside = False
    for poly in polys:
        if _pt_in_ring(pt, poly):
            inside = not inside
    return inside


def _slice_polys(mm_rot, y):
    """在规范系 Y=y 切一刀，返回该截面上的多边形 [(u,v)，...]。

    manifold 的 slice 只沿自身 Z 轴切，所以进门前已经把 (x,y,z)->(x,z,-y) 转好，
    这样切 z=-y 就等价于在规范系里切 Y=y，而截面坐标正好是 (u, v)。
    """
    try:
        cs = mm_rot.slice(-float(y))
    except Exception:                      # noqa: BLE001
        return []
    return [np.asarray(p, float) for p in cs.to_polygons() if len(p) >= 3]


def _ceiling_probe(mesh, ring, T, up=(0.0, 1.0), oblique_deg=45.0,
                   n_samp=16, probe_r=0.09, off=0.14, tw=None, step=0.05):
    """沿 up 边采样，量「上方墙底面」离口边那张面有多深。

    做法：把件在规范系里沿 Y 切成若干平行截面（Y = 墙厚方向），对每个采样点
    取「口边外侧 off 处」那一个 2D 点，看它在哪些截面上落在实心里：
      · 上方墙底面和口边齐平（正常单层悬垂）→ 材料从 Y=0 就有，深度 0；
      · 上方墙底面比口边深（沉台/台阶/锥口）→ 前若干个截面是空的，深度 >0；
      · 该处上方根本没有材料 → 所有截面都空，None。

    与旧版的区别：不打细柱、不做布尔求交、不猜孔环绕向，只做 2D 点在多边形内。
    返回 [(depth, (u,v)), ...]，第三项是采样点坐标；兼容旧调用方，形状不变。
    """
    import manifold3d as m3
    P_ = np.asarray(ring, float)
    if _sarea(P_) < 0:
        P_ = P_[::-1]
    _, t, nout, L = _edge_frames(P_)
    cu = math.cos(math.radians(float(oblique_deg)))
    up = np.asarray(up, float)
    sel = [i for i in range(len(P_)) if float(nout[i] @ up) >= cu and L[i] > 1e-9]
    if not sel:
        return []
    tot = float(sum(L[i] for i in sel))

    mm = m3.Manifold(m3.Mesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces)))
    _rot = np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, -1.0, 0.0, 0.0]], float)   # (x,y,z)->(x,z,-y)
    mm_rot = mm.transform(_rot)
    n_sec = int(max(4, min(400, round(float(T) / max(step, 1e-3)))))
    ys = [(k + 0.5) * float(T) / n_sec for k in range(n_sec)]
    secs = [_slice_polys(mm_rot, y) for y in ys]

    targets = [(k + 0.5) * tot / float(n_samp) for k in range(n_samp)]
    acc = 0.0
    out = []
    for i in sel:
        p0 = P_[i]
        while targets and acc + L[i] >= targets[0]:
            f = (targets.pop(0) - acc) / L[i]
            p = p0 + t[i] * (f * L[i])
            uv = (float(p[0]), float(p[1]))
            # 两侧都试：孔环绕向不定，_edge_frames 的 nout 对孔环可能指向孔内
            qp = p + nout[i] * float(off)
            qm = p - nout[i] * float(off)
            fp = [_in_material(qp, polys) for polys in secs]
            fm = [_in_material(qm, polys) for polys in secs]
            if not any(fp) and not any(fm):
                out.append((None, None, uv))
                continue
            flags = fp if sum(fp) >= sum(fm) else fm
            k0 = flags.index(True)
            k1 = k0
            while k1 + 1 < len(flags) and flags[k1 + 1]:
                k1 += 1
            half = 0.5 * float(T) / n_sec
            out.append((max(0.0, ys[k0] - half), min(float(T), ys[k1] + half), uv))
        acc += L[i]
    return out


def _recess_verdict(probe, tol, T, w_top=0.0):
    """把 _ceiling_probe 的结果判成 (ok, 说明)。ok=False 时不生成薄墙支撑。"""
    if not probe:
        return True, ""
    have = [(y0, y1) for y0, y1, _ in probe if y0 is not None and y1 is not None]
    none_n = len(probe) - len(have)
    if not have or none_n * 2 >= len(probe):
        return False, ("上方没有墙：%d/%d 个采样点沿墙厚方向打空（口上缘外面是空的，没有东西可托）"
                       % (none_n, len(probe)))
    d0 = max(y0 for y0, _ in have)                    # 材料起点离口边那张面有多深
    span = min(y1 - y0 for y0, y1 in have)            # 那处上方材料沿墙厚还有多厚
    if d0 > tol:
        return False, ("上方墙底面比口边深 %.2fmm（> 容差 %.2fmm）：这层悬垂不在口边那一层，"
                       "薄墙支撑托不住它，不生成" % (d0, tol))
    if w_top > 0.0 and span < w_top - 1e-6:
        return False, ("上方墙在那处只剩 %.2fmm 厚（< 张开柱宽 %.2fmm）：托不进墙里，"
                       "钩不住，不生成" % (span, w_top))
    return True, ""


# ============================================================ 报告
def _fmt_warn(warn):
    if not warn:
        return ""
    return "; ".join(str(w) for w in warn)


def _support_line(info):
    sup = (info or {}).get("support") or {}
    up = sup.get("up") or {}
    n_up = up.get("n", 0)
    clear = up.get("clear_max", None)
    clear_s = "-" if clear is None else "%.2f" % float(clear)
    cover = up.get("cover", None)
    cover_s = "-" if cover is None else "%.0f%%" % (100.0 * float(cover))
    return n_up, clear_s, cover_s


# ============================================================ 主流程
def run(args):
    tw = _import_engine()
    t0 = time.time()

    # ---------- 1. 载入 ----------
    path = args.input
    if not os.path.isfile(path):
        raise SystemExit("找不到输入文件：%s" % path)
    mesh = trimesh.load(path, process=True, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit("载入结果不是单一网格（%r）" % type(mesh))
    mesh.merge_vertices()
    if len(mesh.faces) == 0:
        raise SystemExit("网格没有三角面")
    vol = float(mesh.volume)
    if vol < 0:                                     # 法向整体朝内 → 翻转，让法向朝外
        mesh.invert()
        vol = -vol
        print("[载入] 网格体积为负，已整体翻面（保证面法向朝外）")
    print("[载入] %s  面=%d 顶点=%d 体积=%.2f cm³ 水密=%s 绕序一致=%s"
          % (os.path.basename(path), len(mesh.faces), len(mesh.vertices),
             vol / 1000.0, mesh.is_watertight, mesh.is_winding_consistent))
    if not mesh.is_winding_consistent:
        print("[警告] 三角面绕序不一致，面法向可能不可靠（定面结果需人工核对）")

    if not mesh.is_watertight:            # STL 里常见「同位置不同索引」的顶点：先焊再重建拓扑
        h, ndup = _heal(mesh)
        if h.is_watertight and len(h.faces):
            print("[修补] 按位置焊接 %d 个重合顶点、重建拓扑 → 已闭合（面 %d→%d）"
                  % (ndup, len(mesh.faces), len(h.faces)))
            mesh = h
        else:
            from trimesh.grouping import group_rows
            g = group_rows(mesh.edges_sorted)
            cnt = np.array([len(x) for x in g])
            nb, n4 = int((cnt == 1).sum()), int((cnt > 2).sum())
            print("[警告] 网格不闭合：边界边 %d 条、非流形(>2 面)边 %d 条（焊接 %d 个重合顶点后仍不闭合）"
                  % (nb, n4, ndup))
            print("       布尔运算要求闭合实体：这种件会报 Not all meshes are volumes 或切出毛边，"
                  "先在 CAD/切片器里补成实体再跑。")

    # ---------- 2. 定面 ----------
    clusters = _plane_clusters(mesh)
    if args.list_faces:
        print("\n[平面簇] 共 %d 个（按面积降序）" % len(clusters))
        _print_faces(clusters)
        print("\n用 --face-sel N 指定窗口所在面，或 --face-normal x|y|z。")
        return 0
    face, how = _pick_face(clusters, args)
    n_out = np.asarray(face["n"], float)
    d_face = float(face["d"])
    c = n_out * d_face
    print("\n[定面] %s：法向 %s  offset=%.3fmm 面积=%.1fmm² 面数=%d 平面度=%.3f"
          % (how, _fmt_n(n_out), d_face, face["area"], face["n_faces"], face["planar"]))

    # ---------- 3. 定墙厚 ----------
    T, wall_how, cands = _detect_wall(mesh, clusters, face, args)
    if args.verbose and cands:
        print("[定墙厚] 候选平行面（按距离）: %s"
              % ", ".join("%.3fmm(面积%.1f)" % (r["T"], r["area"]) for r in cands[:6]))
    if args.wall is not None:
        print("[定墙厚] 检测到 %.3fmm（%s），按 --wall 覆盖为 %.3fmm"
              % (T, wall_how, args.wall))
        T = float(args.wall)
    else:
        print("[定墙厚] T=%.3fmm（%s）" % (T, wall_how))
    if T <= 0:
        raise SystemExit("墙厚 %.3f 不成立" % T)
    if args.skin is None:
        raise SystemExit("必须给膜厚 --skin（这个参数没有默认值）：按打印机/用途定，"
                         "线宽 0.4 时建议 0.40（1 圈）或 0.80（2 圈）；"
                         "Classic 墙生成器下用整圈数。例：--skin 0.40")
    if args.skin <= 0:
        raise SystemExit("--skin 必须是正数")
    if T < args.skin:
        raise SystemExit("墙厚 %.3fmm < skin %.3fmm：这面墙比薄墙还薄，放不下膜。"
                         "改 --skin 或换面" % (T, args.skin))

    # ---------- 4. 坐标系 ----------
    up_vec, up_label = _up_hint(args)
    if up_vec is None:                              # 自动：挑与面法向夹角最大的世界轴
        Ydir = -n_out
        best = None
        for nm in ("z", "y", "x"):
            cand = _AXES[nm]
            cross = float(np.linalg.norm(np.cross(cand, Ydir)))
            if best is None or cross > best[0]:
                best = (cross, cand, nm)
        up_vec = best[1]
        up_label = "+%s(自动)" % best[2]
    R, X, Y, Z = _build_frame(n_out, up_vec, up_label)
    print("[坐标系] X=u%s  Y=内法向%s  Z=up%s（det(R)=%+.6f）"
          % (_fmt_n(X), _fmt_n(Y), _fmt_n(Z), np.linalg.det(R)))

    # 世界 → 规范
    M_w2c = np.eye(4)
    M_w2c[:3, :3] = R.T
    M_w2c[:3, 3] = -R.T @ c
    M_c2w = np.eye(4)
    M_c2w[:3, :3] = R
    M_c2w[:3, 3] = c
    canon = mesh.copy()
    canon.apply_transform(M_w2c)
    yb = canon.bounds[:, 1]
    # 规范化后：规范系 +Y 从所选面往实体里。这里只查两件真能说明「定面/定厚错位」的事：
    #  · Y<0 有实体 = 所选面朝外的前方还有材料 → 面没选在最外层（盒内壁/沉台会这样）；
    #  · Ymax 还不到墙厚 = 检测到的背面跑出零件外面了。
    # 注意不能要求 Ymax==T：盒体/腔体件的其它壁沿规范系 Y 会继续延伸（实测空心盒 Y∈[0,40]）。
    if float(yb[0]) < -0.5:
        print("[警告] 规范系里 Y 起点 = %.3f < 0：所选面朝外方向的前方还有实体，"
              "所选面可能不是最靠外的面（用 --face-sel / --face-normal 换一个）；"
              "定面/定厚可能错位" % float(yb[0]))
    elif float(yb[1]) < T - 0.5:
        print("[警告] 规范系里 Y 终点 = %.3f < 检测到的墙厚 %.3f：背面可能不在零件的平行面上，"
              "定厚可能错位（可用 --wall 指定）" % (float(yb[1]), T))

    # ---------- 5. 抽孔 ----------
    origin_w = R @ np.array([0.0, T / 2.0, 0.0]) + c        # 中厚平面上的点（世界系）
    loops, n_open, n_pts, n_seg = _section_uv(
        mesh, origin_w, n_out, X, Z, c, args.weld_tol)
    merge_warn = []
    merge_tol = (1.2 * float(args.lw)) if args.merge_thin is None else float(args.merge_thin)
    tag = "、".join("%.2f" % a for a in [merge_tol])
    loops2, reg = _section_regularize(loops, merge_tol, tag)
    if reg is not None:
        if merge_tol > 1e-9:
            print("[规整] 薄壁合并（薄于 %.2fmm 当不存在，= %.2f×线宽）：环 %d→%d、"
                  "轮廓 %d→%d、材料面积 %.2f→%.2f mm²（少 %.2f）"
                  % (merge_tol, merge_tol / max(1e-9, args.lw), reg["rings_before"],
                     reg["rings_after"], reg["contours_before"], reg["contours_after"],
                     reg["area_before"], reg["area_after"], reg["dropped"]))
            if reg["dropped"] > 0.02:
                print("[规整] 删掉了 %.2f mm² 的残留薄壁（切片器本来也印不出来）；"
                      "要保留它们请用 --merge-thin 0 或换一个更小的值" % reg["dropped"])
        else:
            print("[规整] 环合并（不做薄壁删除）：环 %d→%d、轮廓 %d→%d、材料面积 %.2f mm²"
                  % (reg["rings_before"], reg["rings_after"], reg["contours_before"],
                     reg["contours_after"], reg["area_after"]))
        loops = loops2
        if merge_tol > 1e-9 and reg["contours_after"] != reg["contours_before"]:
            merge_warn.append("薄壁合并把尖角削出了倒角（轮廓 %d→%d）：后续判窄可能误伤，"
                              "尖角件请用 --merge-thin 0"
                              % (reg["contours_before"], reg["contours_after"]))
        if merge_tol > 1e-9 and reg["dropped"] > 0.02:
            merge_warn.append("已把薄于 %.2fmm 的残留壁当不存在（删 %.2fmm²）"
                              % (merge_tol, reg["dropped"]))
    holes, outer_idx, deep_idx = _group_holes(loops)
    holes_all = list(holes)
    big = []
    small = []
    for h in holes_all:
        if h["area"] < args.min_hole_area:
            small.append(h)
        else:
            big.append(h)
    print("[截孔] 中厚平面 Y=%.3f（世界点 %s）：线段 %d 条 → 环 %d 个 → 外形环 %d 个、"
          "孔 %d 个（其中面积<%.1fmm² 跳过 %d 个）  [%.2fs]"
          % (T / 2.0, "(%.2f,%.2f,%.2f)" % tuple(origin_w), n_seg, len(loops),
             len(outer_idx), len(holes_all), args.min_hole_area, len(small),
             time.time() - t0))
    if n_open:
        print("[警告] 截面有 %d 条开链（网格在截面处不闭合，可能非水密），已丢弃" % n_open)
    if deep_idx:
        print("[警告] 有 %d 个环嵌套 ≥3 层（岛里还有孔），引擎只支持「孔 + 岛」一层，"
              "这部分可能不对" % len(deep_idx))
    if not big:
        raise SystemExit("没有找到面积 ≥ %.1fmm² 的闭合孔；用 --min-hole-area 调小阈值，"
                         "或确认这个 STL 的孔是通孔（截面才能截到）" % args.min_hole_area)

    for i, h in enumerate(small, 1):
        print("[跳过] 小孔 面积=%.2fmm² < %.1fmm²" % (h["area"], args.min_hole_area))

    # ---------- 6. 引擎参数 ----------
    kwargs = dict(layer=args.layer, lw=args.lw, skin=args.skin, wall=float(T),
                  up=(0.0, 1.0), flare=not args.no_flare,
                  max_wall=float(args.max_wall), top_clear=0.16,
                  post_clear=(0.0 if args.post_clear is None
                              else float(args.post_clear)),
                  tip_clamp_layers=int(args.tip_clamp_layers),
                  tip_blunt=float(args.tip_blunt), tip_blunt_deg=float(args.tip_blunt_deg),
                  membrane_min=float(args.membrane_min))
    if args.max_wall and float(T) > float(args.max_wall) + 1e-9:
        print("[跳过] 墙厚 T=%.3fmm > 上限 %.3fmm：厚墙不需要薄墙支撑（一处悬垂只在薄壁上加一套），"
              "不导出。要强行生成请 --max-wall 0。" % (float(T), float(args.max_wall)))
        raise SystemExit(1)
    if args.span_max is not None:
        kwargs["span_max"] = args.span_max
    if args.gap_side is not None:
        kwargs["gap_side"] = args.gap_side
    if args.gap_min is not None:
        kwargs["gap_min"] = args.gap_min
    if args.post_w is not None:
        kwargs["post_w"] = args.post_w
    if args.post_b is not None:
        kwargs["post_b"] = args.post_b
    spec = tw.WindowSpec(**kwargs)

    # ---------- 7. 逐孔生成 ----------
    report = []
    canon0 = canon.copy()          # 原始件（未改孔），用来量「上方墙底面有多深」
    # ---------- 6b. 瓶颈局部剪除：塞不进膜的窄颈切开成独立孔 ----------
    if not args.no_split_pinch:
        need = 2.0 * spec.gap_up + spec.lw
        expanded = []
        for h in big:
            groups, cuts, nm, dropped = _split_pinches(h["rings"], need)
            if len(groups) <= 1:
                expanded.append({"rings": groups[0], "area": abs(_poly_area(groups[0][0]))})
                continue
            print("[剪除] 口面积 %.1fmm²：%d 处瓶颈（最窄 %.3fmm < %.3fmm = 两侧缝+线宽）"
                  "用弦切开 → 分成 %d 个子口，各自出膜出柱（瓶颈处不铺膜）%s"
                  % (h["area"], len(cuts), (nm if nm is not None else 0.0), need,
                     len(groups), ("；丢掉 %d 个压在瓶颈上的岛" % dropped) if dropped else ""))
            for g in groups:
                expanded.append({"rings": g, "area": abs(_poly_area(g[0]))})
        big = expanded

    for k, h in enumerate(big, 1):
        rings = h["rings"]
        nv = int(sum(len(r) for r in rings))
        head = "孔#%d 环=%d 顶点=%d 面积=%.1fmm²" % (k, len(rings), nv, h["area"])
        if not args.no_recess_check:
            probe = _ceiling_probe(canon0, rings[0], T, (0.0, 1.0),
                                   oblique_deg=spec.oblique_deg, tw=tw)
            w_top_r = (min(spec.flare_wmax, float(T) - spec.lw)
                       if spec.flare else 0.0)
            ok_r, why_r = _recess_verdict(probe, args.recess_tol, float(T), w_top_r)
            if args.verbose:
                print("        [recess] %d 个采样点，材料区间 %s"
                      % (len(probe), ["无" if y0 is None else "%.2f..%.2f" % (y0, y1)
                                     for y0, y1, _ in probe]))
            _ys = [y0 for y0, y1, _ in probe if y0 is not None]
            _flat = (len(_ys) >= max(1, len(probe) // 2)
                     and max(_ys) <= float(args.recess_tol))
            _tc = float(args.top_clear)
            if not ok_r:
                print("[%s] SKIP  %s" % (head, why_r))
                for y0, y1, p in probe:
                    print("        probe (u=%.2f v=%.2f) 材料=%s"
                          % (p[0], p[1], "无" if y0 is None else "Y %.3f..%.3f" % (y0, y1)))
                report.append({"k": k, "rings": len(rings), "nv": nv, "ok": False,
                               "why": why_r, "area": h["area"]})
                continue
        try:
            t1 = time.time()
            import dataclasses as _dc
            _spec_h = _dc.replace(spec, top_clear=_tc)
            cut, fill, info = tw.window_poly(rings, 0.0, T, spec=_spec_h, axis="y")
            dt = time.time() - t1
        except ValueError as e:
            print("[%s] SKIP  引擎拒绝：%s" % (head, e))
            report.append({"k": k, "rings": len(rings), "nv": nv, "ok": False,
                           "why": str(e), "area": h["area"]})
            continue
        except Exception as e:                       # noqa: BLE001
            print("[%s] SKIP  引擎异常：%s: %s" % (head, type(e).__name__, e))
            if args.verbose:
                traceback.print_exc()
            report.append({"k": k, "rings": len(rings), "nv": nv, "ok": False,
                           "why": "%s: %s" % (type(e).__name__, e), "area": h["area"]})
            continue

        if not fill:                                  # 引擎按闸门跳过（不报错、不出件）
            why_s = str(info.get("skipped") or "未生成")
            if why_s == "membrane_too_narrow":
                why_s = ("口太窄：最宽处只有 %.3fmm（< 下限 %.3fmm），塞不进膜与两圈缝"
                         % (float(info.get("membrane_max", float("nan"))),
                            float(info.get("min_membrane", 0.0))))
            elif why_s == "wall_too_thick":
                why_s = ("墙厚 %.3f ≥ 上限 %.3f" % (float(info.get("wall", 0.0)),
                                                 float(info.get("max_wall", 0.0))))
            print("[%s] SKIP  %s" % (head, why_s))
            report.append({"k": k, "rings": len(rings), "nv": nv, "ok": False,
                           "why": why_s, "area": h["area"]})
            continue

        n_up, clear_s, cover_s = _support_line(info)
        gap = float(info.get("gap_side", float("nan")))
        warn = info.get("warn") or []
        print("[%s] gap=%.2f 上柱=%s 净跨=%s 覆盖率=%s  %s"
              % (head, gap, n_up, clear_s, cover_s,
                 "%d 条 warn" % len(warn) if warn else "无 warn"))
        for w in warn:
            print("        warn: %s" % (w,))

        rec = {"k": k, "rings": len(rings), "nv": nv, "ok": True, "area": h["area"],
               "gap": gap, "n_up": n_up, "clear": clear_s, "warn": warn,
               "gap_up": float((info.get("flare") or {}).get("gap", float("nan")))}

        if args.dry_run:
            report.append(rec)
            continue

        snapshot = canon.copy()
        try:
            t1 = time.time()
            if len(rings) > 1:
                # 带岛：岛是 STL 里的真材料，用环形刀（引擎自带的刀会把岛打穿）
                cut = _annular_cut(rings, 0.0, T, spec)
            canon = tw.subtract(canon, cut)
            if args.verbose:
                _c = tw.check(canon)
                print("        [调试] 切刀后：bodies=%d closed=%s vol=%.2f"
                      % (_c["bodies"], _c["closed"], _c["vol_cm3"]))
            canon = tw.add(canon, fill)
            rec["bool_s"] = time.time() - t1
            ck = tw.check(canon)
            rec["closed"] = ck["closed"]
            rec["bodies"] = ck["bodies"]
            print("        布尔+并回：closed=%s bodies=%d vol=%.2fcm³ [%.1fs]"
                  % (ck["closed"], ck["bodies"], ck["vol_cm3"], rec["bool_s"]))
        except Exception as e:                       # noqa: BLE001
            canon = snapshot
            print("        布尔失败，跳过该孔：%s: %s" % (type(e).__name__, e))
            if args.verbose:
                traceback.print_exc()
            rec["ok"] = False
            rec["why"] = "布尔失败: %s: %s" % (type(e).__name__, e)
        report.append(rec)
    ok = [r for r in report if r["ok"]]
    bad = [r for r in report if not r["ok"]]
    if bad:
        print("\n[跳过] %d 个孔：%s" % (len(bad), "；".join(
            "孔#%d %s" % (r["k"], r["why"]) for r in bad)))
    print("[统计] 孔 %d 个：成功 %d、跳过 %d（另小孔跳过 %d）"
          % (len(big), len(ok), len(bad), len(small)))

    # ---------- 8. 导出 ----------
    if args.dry_run:
        print("[dry-run] 不做布尔、不导出（用时 %.2fs）" % (time.time() - t0))
        return 0
    if not ok:
        raise SystemExit("没有任何孔成功生成薄墙，不导出")

    back = canon.copy()
    back.apply_transform(M_c2w)
    back.merge_vertices()
    if args.verbose:
        _c = tw.check(back)
        print("[调试] 焊接后：closed=%s nonmanifold=%s watertight=%s 顶点=%d"
              % (_c["closed"], _c["nonmanifold"], _c["watertight"],
                 len(back.vertices)))
    back, ndup = _heal(back)
    if args.verbose:
        _c = tw.check(back)
        print("[调试] 修复后：closed=%s nonmanifold=%s watertight=%s 顶点=%d"
              % (_c["closed"], _c["nonmanifold"], _c["watertight"],
                 len(back.vertices)))
    if ndup:
        print("[修复] 焊掉 %d 个位置重复的顶点（不修的话导出后会有非流形边）" % ndup)
    back, rinfo = _regularize(back)
    if rinfo["n_in"] > 1:
        drops = ("（%s cm³）" % ", ".join("%g" % v for v in rinfo["drop"])
                 if rinfo["drop"] else "")
        print("[修复] 布尔结果有 %d 个连通体：丢 %d 个零体积退化解片%s，其余自并成一片"
              % (rinfo["n_in"], len(rinfo["drop"]), drops))
    back, ndup2 = _heal(back)
    if ndup2:
        print("[修复] 自并后重焊 %d 个位置重复的顶点" % ndup2)
    out = args.out or (os.path.splitext(path)[0] + "_thinwall.stl")
    res = tw.clean_export(back, out)
    if not (res.get("closed") and res.get("watertight")):
        # thinwall.clean() 里 `merge_vertices(digits_vertex=6)` 会把 manifold3d 布尔输出里
        # 「位置完全重复、索引不同」的顶点焊成一个 → 变成「一条边 4 个面」的非流形边。
        # 这不是本脚本造成的：引擎自带的圆孔算例单独跑 window_poly+clean_export 也能复现
        # （closed=True、体积精确，只是严格 watertight 校验不过）。thinwall.py 不动，
        # 这里自己再修一遍：去退化片 → 按位置去重 → manifold3d 重建拓扑 → 先写临时文件
        # 复核，只有确实更好才替换正式输出，绝不把 clean_export 的好结果弄坏。
        tmp = out + ".fix.stl"
        try:
            cand = back
            cand, _ = _regularize(cand)
            cand, nd2 = _heal(cand)
            cand.export(tmp)
            ck = tw.check(trimesh.load(tmp, process=True, force="mesh"))
            better = bool(ck.get("closed")
                          and (ck.get("watertight") or not res.get("closed")))
            if better and len(cand.faces):
                os.replace(tmp, out)
                res.update(ck)
                res["reload_watertight"] = bool(ck["watertight"])
                print("[修复] thinwall.clean() 之后不严格闭合：重焊 %d 个重复顶点 + 丢退化"
                      "解片后重新导出（closed=True、watertight=%s）"
                      % (nd2, ck["watertight"]))
            elif os.path.exists(tmp):
                os.remove(tmp)
        except Exception as e:                       # noqa: BLE001
            print("[修复] 失败（保留 clean_export 的原始结果）：%s: %s"
                  % (type(e).__name__, e))
            if args.verbose:
                traceback.print_exc()
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    print("\n[导出] %s" % res.get("path", out))
    print("       closed=%s bodies=%s vol=%.2fcm³ nonmanifold=%s holes=%s "
          "watertight=%s reload_watertight=%s"
          % (res.get("closed"), res.get("bodies"), res.get("vol_cm3"),
             res.get("nonmanifold"), res.get("holes"), res.get("watertight"),
             res.get("reload_watertight")))
    if not res.get("closed"):
        print("[警告] 导出网格有边界边（不闭合），切片器可能报错")
    elif not res.get("watertight"):
        print("[说明] 严格 watertight=False、但 closed=True（无边界边、无洞）：这是引擎已知的"
              "「连接件与膜共面时面被合并、三角化出重复面」现象（只有 4 面边），"
              "体积/体数/闭合都正确，切片器按闭合体处理。")
    print("[完成] %.2fs" % (time.time() - t0))
    return 0 if res.get("closed") else 2


# ============================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(
        prog="stl_window.py",
        description="从 STL 抽出镂空孔边界，调 thinwall 引擎在孔上生成薄墙，布尔回网格后导出 STL。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 先看有哪些平面可当窗口面
  python stl_window.py in.stl --list-faces
  # 自动定面 + 生成 + 导出（层高 0.08）
  python stl_window.py in.stl --out out.stl --layer 0.08
  # 指定面与打印向上方向
  python stl_window.py in.stl --out out.stl --face-sel 0 --up +z
  # 只抽环和报告，不做布尔
  python stl_window.py in.stl --dry-run
""")
    ap.add_argument("input", help="输入 STL 路径")
    ap.add_argument("--out", default=None, help="输出 STL 路径（默认 <输入>_thinwall.stl）")
    ap.add_argument("--layer", type=float, default=0.08, help="层高 mm（默认 0.08）")
    ap.add_argument("--lw", type=float, default=0.4, help="线宽 mm（默认 0.4）")
    ap.add_argument("--no-split-pinch", action="store_true",
                    help="不做瓶颈剪除：孔环上净距 < 2×上缘缝+线宽 的窄颈不被弦切开，"
                         "这类孔会被引擎整体拒绝")
    ap.add_argument("--merge-thin", type=float, default=0.0, metavar="MM",
                    help="把薄于该厚度的残留壁当不存在（**默认 0 = 不删材料**，只合并重叠环）。"
                         "给正值是形态学开运算，会在 36° 这类尖角上削出细小倒角（实测 0.058），"
                         "尖角件务必留 0；只有确实有薄于 1.2×线宽 的残留壁且不需要保尖角时才开")
    ap.add_argument("--skin", type=float, default=None,
                    help="膜厚 mm（生成薄墙时必填、没有默认值：每次按打印机/用途定；"
                         "建议 lw~2×lw，Classic 墙生成器用整圈数 0.40 / 0.80）")
    ap.add_argument("--max-wall", type=float, default=5.0,
                    help="墙厚上限 mm（默认 5.0）：量到的墙厚 ≥ 该值就不做薄墙支撑，直接不导出；0 = 关掉这条闸门")
    ap.add_argument("--top-clear", type=float, default=0.16, metavar="MM",
                    help="上缘张开柱顶端离上方墙面的间隙 mm（**固定 0.16**，0.08 层高下正好 2 层）："
                         "柱顶一律不许长进墙里，远端停在离墙面该处")
    ap.add_argument("--post-clear", type=float, default=None, metavar="MM",
                    help="上缘张开柱**根部**（与膜结合那一端）离膜的净空 mm；默认 2×层高"
                         "（0.08 层高 → 0.16）。0 = 老口径（坡根压进膜里）。"
                         "根部退开后柱子只与墙相连，若因此掉件会自动回退并报警。")
    ap.add_argument("--tip-clamp-layers", type=int, default=2, metavar="N",
                    help="尖角横桥占几个层高（默认 2；桥高 = N×layer）")
    ap.add_argument("--tip-blunt", type=float, default=0.8, metavar="MM",
                    help="膜的尖角钝化宽度：内角特别尖的膜尖会被削成这么宽的平口"
                         "（默认 0.8 = 2 条 0.4 线宽；尖角处每层放不下一条线，切片器会丢掉）。"
                         "给 0 关掉钝化、保留原来的尖尖。")
    ap.add_argument("--tip-blunt-deg", type=float, default=70.0, metavar="DEG",
                    help="内角小于该值（默认 70°）的膜角才钝化；90° 直角本来就能打、不处理，圆滑曲线的离散顶点也不受影响。")
    ap.add_argument("--wall", type=float, default=None,
                    help="覆盖自动检测的墙厚 mm（默认自动）")
    ap.add_argument("--face-normal", default="auto",
                    help="窗口面法向：auto|x|y|z（可 +y/-y 指定朝向，默认 auto=最大平面簇）")
    ap.add_argument("--face-sel", type=int, default=None,
                    help="用 --list-faces 的序号直接指定窗口面")
    ap.add_argument("--up", default="+z",
                    help="打印向上方向（世界系）：+z|z|y|x|…；- 表示自动挑（默认 +z）")
    ap.add_argument("--min-hole-area", type=float, default=20.0,
                    help="最小孔面积 mm²，更小的孔跳过（默认 20）")
    ap.add_argument("--list-faces", action="store_true",
                    help="只列出平面簇（法向/面积/平面上一点）就退出")
    ap.add_argument("--dry-run", action="store_true",
                    help="只抽环 + 跑引擎出报告，不做布尔、不导出")
    ap.add_argument("--weld-tol", type=float, default=1e-4,
                    help="截面点焊接容差 mm（默认 1e-4）")
    ap.add_argument("--recess-tol", type=float, default=0.05,
                    help="上方墙底面比口边深超过多少 mm 就不再生成薄墙支撑（默认 0.05）")
    ap.add_argument("--no-recess-check", action="store_true",
                    help="关闭「上方墙底面比口边深」检查（强行生成）")
    # 引擎参数透传（一般不用动）
    ap.add_argument("--span-max", type=float, default=None, help="上缘允许净跨 mm")
    ap.add_argument("--membrane-min", type=float, default=0.4, metavar="MM",
                    help="膜最窄处下限 mm（净宽 − 两侧缝），默认 0.4，不跟线宽变")
    ap.add_argument("--gap-side", type=float, default=None, help="非张开侧缝宽 mm")
    ap.add_argument("--gap-min", type=float, default=None, help="缝宽下限 mm")
    ap.add_argument("--post-w", type=float, default=None, help="上缘张开柱宽 mm")
    ap.add_argument("--post-b", type=float, default=None, help="下缘直条柱宽 mm")
    ap.add_argument("--no-flare", action="store_true", help="上缘不张开（用 V 形楔）")
    ap.add_argument("--verbose", action="store_true", help="打印更多诊断")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    raw = list(argv) if argv is not None else list(sys.argv[1:])
    try:
        return run(args)
    except SystemExit as e:
        msg = "" if isinstance(e.code, int) or e.code is None else str(e.code)
        # 自动定面挑的是「面积最大的平面」。一件上有好几张平行面时，孔常常在面积稍小的那张上
        # （例如 50mm 立方壳：五角星开在 +Y 面 2212mm²，而 ±X/±Y 的素面各 2500mm²）。
        # 这种「自动面上没孔」的情况就挨个换面再试，别让使用者先去读 --list-faces。
        if (args.face_sel is None and args.face_normal in (None, "auto", "")
                and "没有找到面积" in msg and not args.dry_run):
            for i in range(1, 8):
                print("\n[自动] 面积最大的那张面上截不到孔，换 --face-sel %d 再试 ……" % i)
                try:
                    return run(build_parser().parse_args(raw + ["--face-sel", str(i)]))
                except SystemExit as e2:
                    m2 = "" if isinstance(e2.code, int) or e2.code is None else str(e2.code)
                    if "越界" in m2:
                        break
                    if "上限" in m2:
                        raise                      # 墙厚闸门这类真结论，不要挨个面重试
                    continue
        raise


if __name__ == "__main__":
    sys.exit(main())
