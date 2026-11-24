"""小层高回归：0.08 / 0.12 / 0.16 / 0.2 mm 层高 × 六种异形口。

为什么单独一套：默认 `layer=0.2` 的 demo-poly 覆盖不到 0.08。层高一变，三个
由层高推出来的量会同时变（缝宽 3×layer、上缘缝 4×layer、张开柱坡度 30° 的
`run`），而 0.08 时 `3×layer=0.24` 已经低于线宽 0.4，必须靠 `gap_min` 抬到
0.4 才切得出来——所以小层高不是"把 layer 改小"就完事，得按下面这几条断言验。

用法：
    python test_small_layer.py                 # 全矩阵，只在终端打表
    python test_small_layer.py --out out/       # 另外导出 0.08 的 STL
    python test_small_layer.py --layers 0.08    # 只跑一档

判据（任一不满足即返回码 1）：
  1. `check()` 的 closed 为真、bodies 恰为 1（膜和墙是同一张闭合面）；
  2. 缝宽 ≥ max(3×layer 下限, lw)：低于 lw 切片器必然填实，膜和墙连成一块；
  3. 上缘净跨 ≤ span_max：这是"上方墙首层要桥多长"的直接量；
  4. 上缘单颗柱搭接面积 ≥ 2×lw×skin：小于这个数锚不住，膜首层会被拉翻。

依赖：manifold3d / numpy（`pip install manifold3d numpy`）。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thinwall as tw  # noqa: E402

LAYERS = (0.08, 0.12, 0.16, 0.20)


def shapes():
    """六种口：直角矩形 / 圆弧 / 工字（含岛效应）/ 拱顶 / 五角星（尖角+凹角）/ 异型凹口。"""
    return [("矩形", tw._rings_rect(25, 175, 30, 260)),
            ("圆形", tw._rings_circle()),
            ("工字形", tw._rings_i()),
            ("拱顶", tw._rings_arch()),
            ("五角星", tw._rings_star()),
            ("异型带凹口", tw._rings_notch())]


def build(rings, spec, panel_w=200.0, panel_h=300.0, thick=3.0):
    cut, fill, info = tw.window_poly(rings, 0.0, thick, spec=spec, axis="y")
    panel = tw.subtract(tw._panel(panel_w, panel_h, thick), cut)
    panel = tw.add(panel, fill)
    return panel, info


def run_layer(layer, out_dir=None, verbose=True):
    spec = tw.WindowSpec(layer=layer, lw=0.4, skin=0.45, wall=3.0, up=(0.0, 1.0))
    g_floor = max(spec.g0, spec.lw)
    rows, bad = [], []
    for tag, rings in shapes():
        try:
            mesh, info = build(rings, spec)
        except ValueError as exc:
            rows.append((tag, None, None, None, None, None, "拒绝：%s" % str(exc)[:40]))
            bad.append((tag, "拒绝了"))
            continue
        res = tw.check(mesh)
        up = info["support"].get("up", {})
        ok = (res["closed"] and res["bodies"] == 1
              and info["gap_side"] >= g_floor - 1e-6
              and up.get("clear_max", 0.0) <= spec.span_max + 5e-3
              and up.get("contact_mm2", 0.0) >= 2.0 * spec.lw * spec.skin)
        if not ok:
            bad.append((tag, "closed=%s bodies=%s 缝=%.3f 墙端跨=%.3f" %
                        (res["closed"], res["bodies"], info["gap_side"],
                         up.get("clear_max", -1.0))))
        rows.append((tag, info["gap_side"], info["flare"]["gap"], up.get("n", 0),
                     up.get("clear_max", 0.0), up.get("contact_mm2", 0.0),
                     "OK" if ok else "FAIL"))
        if out_dir:
            tw.clean_export(mesh, os.path.join(out_dir, "L%03d_%s.stl" % (round(layer * 1000), tag)))
    if verbose:
        print("layer=%.2f  lw=0.40  skin=0.45  span_max=%.1f" % (layer, spec.span_max))
        print("  %-10s %-6s %-6s %-5s %-8s %-9s %s" %
              ("形状", "缝侧", "缝上", "上柱", "净跨", "搭接mm2", "结果"))
        for tag, a, b, c, d, e, f in rows:
            if a is None:
                print("  %-10s %-6s %-6s %-5s %-8s %-9s %s" % (tag, "-", "-", "-", "-", "-", f))
            else:
                print("  %-10s %-6.3f %-6.3f %-5d %-8.3f %-9.2f %s" % (tag, a, b, c, d, e, f))
    return rows, bad


def run_too_narrow(layer=0.08):
    """口太窄闸门：整口两侧缝之间最宽处 < min_membrane(2.0) 时**不生成**（返回空件、不报错）。

    1.2mm 窄槽在 0.6 缝下只剩 0.04 宽的膜——几何闭合、片数对，但打不出来，所以整口跳过。
    """
    lim = tw.WindowSpec().min_membrane
    print("\n口太窄闸门（layer=%.2f，min_membrane=%.1f）：" % (layer, lim))
    print("  %-8s %-8s %-6s %s" % ("净宽", "最宽膜", "结果", "说明"))
    bad = []
    for w in (1.2, 2.4, 3.0, 3.4):
        spec = tw.WindowSpec(layer=layer, lw=0.4, skin=0.45, wall=3.0, up=(0.0, 1.0))
        rings = [[(60.0, 20.0), (60.0 + w, 20.0), (60.0 + w, 120.0), (60.0, 120.0)]]
        cut, fill, info = tw.window_poly(rings, 0.0, 3.0, spec=spec, axis="y")
        sk = info.get("skipped")
        mw = info.get("membrane_max", float("nan"))
        if w < 3.2 - 1e-9:
            ok = (sk == "membrane_too_narrow")
            why = "按预期跳过" if ok else "该跳过却生成了"
        else:
            ok = not sk
            why = "正常生成" if ok else "不该跳过却跳过（%s）" % sk
        if not ok:
            bad.append(("净宽 %.1f" % w, why))
        print("  %-8.1f %-8.3f %-6s %s" % (w, mw, "跳过" if sk else "生成", why))
    return bad


def main():
    ap = argparse.ArgumentParser(description="小层高（0.08 起）薄墙回归")
    ap.add_argument("--out", default=None, help="导出 STL 的目录（给了就导出每档每形状）")
    ap.add_argument("--layers", default=",".join("%.2f" % v for v in LAYERS))
    args = ap.parse_args()
    layers = [float(x) for x in args.layers.split(",") if x.strip()]
    out_dir = os.path.abspath(args.out) if args.out else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    all_bad = []
    for layer in layers:
        rows, bad = run_layer(layer, out_dir)
        all_bad += [("layer=%.2f/%s" % (layer, t), m) for t, m in bad]
        print("")
    all_bad += run_too_narrow(layers[0])
    print("")
    if all_bad:
        print("FAIL %d 项：" % len(all_bad))
        for t, m in all_bad:
            print("  - %s：%s" % (t, m))
        return 1
    print("全部通过（%d 档层高 × %d 形状 + 口太窄闸门 4 档）" %
          (len(layers), len(shapes())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
