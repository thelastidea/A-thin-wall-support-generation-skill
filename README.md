# fdm-thinwall-window

给 FDM 3D 打印盒体/面板的镂空口生成「**居中薄墙（膜）+ 支撑柱 + 侧向连接件**」结构：把通孔窗口改成一片悬浮在壁厚中面的薄膜，四周留缝，靠连接件把膜焊回主体。

**它按"支撑"来设计，不是按"封口"**：膜是被撑住的那片薄板；下缘柱是它的脚（膜第一条层带只落在柱顶上、只桥 ≤3mm），上缘 30° 张开柱是上方墙底的托（顶端搭接面 `w_top×2.0`，沿边不加宽，靠柱数把净跨压到 ≤3mm），侧边薄墙片是防倒伏的撑（每片只占**一个层高**，默认 `rib_w = layer`、柱心自动压到层带中心，不在缝里砌出多层高的墙）。**膜不密封**——四周的缝是通的，别当封口用。

> **⚠️ 测试阶段**：参数口径在持续实测迭代，不同版本行为可能变化。任何生成件在正式打印前，请**先看预览图、再在切片器中逐层确认**（重点：缝隙是否被间隙填充焊死、起膜边是否下垂粘连）。打印效果与撕脱体验请自行验证，风险自担。

## 能干什么

- **只在墙厚 < 5mm 时生成**（`max_wall`，按沿墙厚方向的欧氏距离）；从 STL 改件时还会先量「上方墙底面」——比口边更深（沉台/台阶/锥口）就直接跳过不出件（`--recess-tol`，默认 0.05mm）；一处悬垂只加一套：膜 + 上缘张开柱 + 下缘直条柱 + 起膜边竖直柱 + 侧边薄墙片，合称**薄墙支撑**。
- 上缘张开柱顶端离上方墙面留 **0.16mm 顶隙**（`top_clear` 写死，层高变了不跟着变）：托住悬垂面但不与它熔死。

- **任意闭合折线轮廓**：矩形、圆形、工字形（凹角）、异型凹口、拱顶、五角星，支持**内环/岛**
- **曲线与变宽口**：椭圆、圆、blob、齿轮、泪滴、波浪槽、新月都实测闭合单片；`corner_deg` 判真拐角（圆弧不误判成拐角）、`var_gap` 逐段压缝（窄端膜与墙粘牢、不会碎成条）
- **小层高可用**：层高从 **0.08** 起；缝宽 `max(3×层高, 1.5×线宽)` / 上缘 `max(4×层高, 缝侧+层高)`——层高掉下去时由线宽托底，不会出现"缝比一条线还窄被切片器填死"
- **四类件自动布置**：上缘张开柱（30° 跟缝法向）/ 下缘竖直柱 / 侧边水平薄墙片 / 尖角件（默认关），净跨 ≤3mm，不骑拐角、凹角留空
- **顶部接触面只在墙厚方向**：`w_top` 2.6 × `post_w` 2.0 = 5.2mm² 单颗搭接，沿边不加宽（要给更大的托面就多放柱，净跨恒 ≤3mm）
- **两道闸门**：墙厚 ≥5mm 不出件；口内最宽处 <2mm（`min_membrane`）也不出件——都不报错，返回空 fill + `info["skipped"]`
- **打印可行性复核**：首层下垂、间隙填充合并、窄通道漏膜（净宽不足直接报错，不静默通过）
- **校验与出图**：密封性 + 单连通体校验、1:1 正投影与剖切预览、可打印 STL 导出

## 目录结构

```
fdm-thinwall-window/
├── SKILL.md                     # 技能说明：结构模型、参数口径、布点规则、校验
├── README.md                    # 本文件
├── references/
│   └── parameters.md            # 参数表、推导、实测矩阵、24 条踩坑、切片设置
└── scripts/
    ├── thinwall.py              # 核心库：膜内缩、连接件布置、布尔与校验、回归自检
    ├── preview.py               # 正投影/剖切预览（numpy + PIL + matplotlib）
    ├── stl_window.py            # 从 STL 直接抽孔边界并改件（薄壁悬浮窗口）
    └── test_small_layer.py      # 小层高回归：4 档层高 × 6 种口 + 口太窄闸门 4 档
```

## 依赖

Python 3，几何核心依赖 [manifold3d](https://github.com/elalish/manifold)，预览另需 numpy / pillow / matplotlib：

```bash
pip install manifold3d numpy pillow matplotlib
```

## 使用方法

### 方式一：作为 Agent 技能（推荐）

把整个目录放进 Agent 的 skills 目录（例如项目的 `.dsh/skills/`、豆包的 `.user_skills/`），然后直接用自然语言下指令，例如"给这个镂空口加薄墙""窗口改成薄壁悬浮、靠连接件拉住""层高 0.08 也要打得出来"。

每次开工需要给定四个参数（没有默认值，刻意不设）：

| 参数 | 含义 |
|---|---|
| `layer` | 层高，决定缝宽与弱连接空档（最小 0.08） |
| `lw` | 线宽，决定可打印缝宽下限、张开封顶、薄墙片宽 |
| `skin` | 膜厚 |
| `up_dir` | 口在平面上哪边朝"打印向上"（决定分区） |

### 方式二：作为 Python 库直接调用

```python
import sys; sys.path.insert(0, r"<本仓库路径>/scripts")
import thinwall as tw

spec = tw.WindowSpec(layer=0.08, lw=0.4, skin=0.45, wall=3.0, up=(0, 1))
rings = [[(25,30),(175,30),(175,260),(25,260)]]   # 外环；有岛再加内环

cut, fill, info = tw.window_poly(rings, 0.0, 3.0, spec=spec, axis="y")
solid = tw.subtract(solid, cut)
solid = tw.add(solid, fill)

print(info["warn"])        # 先读告警（窄缝/净跨超标/上缘无柱/起膜边提醒/窄端不放柱）
print(info["support"])     # 每区柱数、净跨、覆盖率、单颗搭接面积
print(tw.check(solid))     # 须 closed=True 且 bodies==1
tw.clean_export(solid, "part.stl")
```

### 预览与自检

```bash
# 出 4 张 1:1 正投影/剖切图：整件、角部、中层剖、跨缝剖
python scripts/preview.py part.stl --out <目录> --normal y --up z

# 回归自检：矩形/圆形/工字形/异型/岛环/拱顶/五角星等 12 例（含 2 例负例）
python scripts/thinwall.py demo-poly --out <目录>

# 小层高回归：0.08/0.12/0.16/0.2 × 6 种口 + 口太窄闸门四档；非零返回码即失败
python scripts/test_small_layer.py --out <目录>

# 从 STL 直接改件：列出候选平面 → 选面 → 抽孔边界 → 生成薄墙 → 导出
python scripts/stl_window.py part.stl --list-faces
# 薄壁阈值可调（默认 1.2×线宽）；不想剪窄颈就 --no-split-pinch
python scripts/stl_window.py part.stl --out part_wall.stl --face-sel 0 --layer 0.08 --merge-thin 0.6
python scripts/stl_window.py part.stl --out <目录> --face-sel 0 --layer 0.08 --up z
# 上方墙底面比口边深就跳过：--recess-tol 0.05（默认）；强行生成：--no-recess-check
```

## 更多文档

- 结构模型、分区与布点规则、校验清单：见 [SKILL.md](SKILL.md)
- 参数表、推导、实测矩阵、踩坑与切片设置：见 [references/parameters.md](references/parameters.md)

## 许可证

沿用上游 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)（本目录未随附 LICENSE 文件）：可自由使用与修改，但衍生作品须以相同许可证开源；将本代码用于网络服务也须公开源码。
