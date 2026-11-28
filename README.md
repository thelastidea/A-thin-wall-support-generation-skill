# fdm-thinwall-window

给 FDM 3D 打印盒体/面板的镂空口生成「**居中薄墙（膜）+ 支撑柱 + 侧向连接件**」结构：把通孔窗口改成一片悬浮在壁厚中面的薄膜，四周留缝，靠连接件把膜焊回主体。

> **⚠️ 测试阶段**：代码全是AI写的。参数口径在持续实测迭代，不同版本行为可能变化。任何生成件在正式打印前，请**先看预览图、再在切片器中逐层确认**（重点：缝隙是否被间隙填充焊死、起膜边是否下垂粘连）。打印效果与撕脱体验请自行验证，风险自担。

## 能干什么

- **只在墙厚 < 5mm 时生成**（按沿墙厚方向的欧氏距离就是直线距离）；从上方墙底面比口边更深（沉台/台阶/锥口）跳过不出件（默认 0.05mm）；目前只能在镂空处于同一平面上生成。
- 上缘张开柱顶端离上方墙面留 **0.16mm 顶隙**

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
作为 Agent 技能
把整个目录放进 Agent 的 skills 目录（例如项目的 `.dsh/skills/`、豆包的 `.user_skills/`），然后直接用自然语言下指令，例如"给这个镂空口加薄墙""窗口改成薄壁悬浮、靠连接件拉住""层高 0.08 也要打得出来"。
每次开工需要给定四个参数
| 参数 | 含义 |
|---|---|
| `layer` | 层高，决定缝宽与弱连接空档（最小 0.08） |
| `lw` | 线宽，决定可打印缝宽下限、张开封顶、薄墙片宽 |
| `skin` | 膜厚 |(0，4喷嘴选0.45，0.2喷嘴选0.4）
| `up_dir` | 口在平面上哪边朝"打印向上"（决定分区） 把stl文件给进去|
## 更多文档

- 结构模型、分区与布点规则、校验清单：见 [SKILL.md](SKILL.md)
- 参数表、推导、实测矩阵、踩坑与切片设置：见 [references/parameters.md](references/parameters.md)

## 许可证

沿用上游 [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)（本目录未随附 LICENSE 文件）：可自由使用与修改，但衍生作品须以相同许可证开源；将本代码用于网络服务也须公开源码。
