fdm-thinwall-window
给 FDM 3D 打印盒体/面板的镂空口生成「居中薄墙（膜）+ 连接件」结构：把通孔窗口改成一片悬浮在壁厚中面的薄膜，四周留缝、靠连接件把膜焊回主体，打印后像撕邮票齿孔一样把膜整片撕掉。

⚠️ 本技能仍处于测试阶段：参数口径在持续实测迭代，不同版本的行为可能变化。任何生成件在正式打印前，请先看预览图、再在切片器中逐层确认（重点：缝隙是否被间隙填充焊死、起膜边是否下垂粘连）。打印效果与撕脱体验请自行验证，风险自担。

能干什么
任意闭合折线轮廓：矩形、圆形、工字形（凹角）、异型凹口、拱顶、五角星，支持内环/岛
缝宽自动分区：上缘固定 4×层高，其余 3×层高，内缩一次成型、保尖角
三类连接件自动布置：上缘 30° 张开柱（跟缝法向）/ 下缘竖直柱 / 侧边水平薄墙片，净跨 ≤ 3 mm，不骑拐角、凹角留空
打印可行性复核：首层下垂、间隙填充合并、窄通道漏膜（净宽不足直接报错，不静默通过）
校验与出图：密封性 + 单连通体校验、1:1 正投影与剖切预览、可打印 STL 导出
目录结构
fdm-thinwall-window/ ├── SKILL.md # 技能说明：全部参数口径、结构定义、校验规则 ├── references/ │ └── parameters.md # 参数细节与实测踩坑记录 └── scripts/ ├── thinwall.py # 核心库：膜内缩、连接件布置、布尔与校验、回归自检 └── preview.py # 正投影/剖切预览（纯 numpy + PIL + matplotlib）

依赖
Python 3，几何核心依赖 manifold3d，预览另需 numpy / pillow / matplotlib：

bash pip install manifold3d numpy pillow matplotlib

使用方法
方式一：作为 Agent 技能（推荐）
把整个目录克隆到你 Agent 的 skills 目录下（例如豆包的 .user_skills/），然后直接用自然语言下指令，例如"给这个镂空口加薄墙""窗口改成薄壁悬浮、靠连接件拉住"。

每次开工需要给定四个参数（没有默认值，刻意不设）：

参数	含义
layer	层高，决定缝宽（3×/4×层高）
lw	线宽，决定张开封顶与最小可打印缝
skin	膜厚
up_dir	口在平面上哪边朝"上"（决定上下缘分区）
方式二：作为 Python 库直接调用
```python import sys; sys.path.insert(0, r"<本仓库路径>/scripts") import thinwall as tw

spec = tw.WindowSpec(layer=0.2, lw=0.4, skin=0.45, wall=3.0, up=(0, 1)) rings = [[(25,30),(175,30),(175,260),(25,260)]] # 外环；有岛再加内环

cut, fill, info = tw.window_poly(rings, 0.0, 3.0, spec=spec, axis="y") solid = tw.subtract(solid, cut) solid = tw.add(solid, fill)

print(tw.check(solid)) # 须 closed=True 且 bodies=1 tw.clean_export(solid, "part.stl") ```

预览与自检
```bash

出 4 张 1:1 正投影/剖切图：整件、角部、中层剖、跨缝剖
python scripts/preview.py part.stl --out <目录> --normal y --up z

回归自检：矩形/圆形/工字形/异型/岛环/拱顶/五角星等 13 例
python scripts/thinwall.py demo-poly --out <目录> ```

更多文档
完整参数口径、连接件布置规则、校验清单：见 SKILL.md
实测踩坑与切片设置建议：见 references/parameters.md
许可证
AGPL-3.0：可自由使用与修改，但衍生作品须以相同许可证开源；将本代码用于网络服务也须公开源码。
