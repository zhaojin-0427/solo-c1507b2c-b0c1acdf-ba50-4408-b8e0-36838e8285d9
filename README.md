# Music-Box Cylinder Pin Arranger

为机械音乐盒制作者提供的本机 API：把乐谱音符换算成滚筒上的植钉坐标，检查可制造性冲突，
必要时搜索可制造的替代方案，并把选定方案冻结为不可变版本，输出可打印的 SVG 滚筒展开图。

- 纯本机运行：Python + FastAPI + Pydantic + SQLite，不接任何外部服务
- 完全确定性：同一输入重复计算（包括跨进程、跨机器）得到逐字节相同的结果与哈希

## 安装与启动

```bash
pip install -r requirements.txt
uvicorn musicbox.main:app --reload          # http://127.0.0.1:8000
```

- 交互式文档：`http://127.0.0.1:8000/docs`
- 数据库文件：默认 `./musicbox.db`，可用环境变量 `MUSICBOX_DB` 指定
- 运行测试：`python3 -m pytest`

## 工作流

```
POST /api/arrangements/check    1. 换算坐标 + 诊断（定位每类问题的首个位置）
POST /api/arrangements/search   2. 排不下时：搜索可制造方案并排序
POST /api/versions              3. 把选定方案冻结为不可变版本（幂等）
GET  /api/versions/{id}         4. 读取版本：钉坐标、冲突余量、输入哈希、SVG
POST /api/versions/{id}/recompute  5. 重算校验：结果必须与冻结时完全一致
```

动平衡（基于已冻结的植钉版本）：

```
POST /api/balance/analyze       1. 不平衡分析：质心偏移、静不平衡、力偶、轴承载荷、逐钉贡献
POST /api/balance/search        2. 从配重库存搜索校正配重并排序
POST /api/balance/plans         3. 把选定配重方案冻结为不可变版本（幂等）
GET  /api/balance/plans/{id}    4. 读取方案：基线/最终不平衡、逐钉贡献、SVG
POST /api/balance/plans/{id}/recompute  5. 重算校验
```

动力试算（基于已冻结的植钉版本）：

```
POST /api/dynamics/trials                       1. 建立不可变试算版本（发条/齿轮/惯量/调速器/拨簧耗能）
POST /api/dynamics/trials/{id}/simulate         2. 固定步长积分滚筒角速度，定位首次停转/超速/节拍漂移
POST /api/dynamics/trials/{id}/search           3. 搜索预紧圈数 × 调速器系数 × 飞轮惯量候选并排序
POST /api/dynamics/plans                        4. 冻结选定方案（来源快照 + 曲线 + 步长 + 哈希，幂等）
GET  /api/dynamics/plans/{id}                   5. 读取方案：参数、场景、完整仿真结果
POST /api/dynamics/plans/{id}/recompute         6. 重算校验
```

装配校准（基于已冻结的植钉版本）：

```
POST /api/calibration/batches                1. 建立校准批次（采集中）：跳动/轴承/钉高/簧片测量，拟合轴线
GET  /api/calibration/batches/{id}           2. 读取批次：拟合参数、逐钉接触/偏差/拨入量/余量、首违规定位
POST /api/calibration/batches/{id}/search    3. 在锁定约束内搜索轴承垫片 × 音梳横移 × 高度方案并排序
POST /api/calibration/batches/{id}/confirm   4. 确认选定调整 → 冻结校准方案（幂等），批次转为已确认
GET  /api/calibration/plans/{id}             5. 读取已确认方案
POST /api/calibration/plans/{id}/recompute   6. 重算校验
```

### 1. 检查（check）

提交乐谱与机构参数（完整示例见 `examples/check_request.json`）：

```bash
curl -s -X POST localhost:8000/api/arrangements/check \
  -H 'Content-Type: application/json' \
  -d @examples/check_request.json | jq
```

输入：

| 字段 | 含义 |
| --- | --- |
| `notes[]` | `id`、`beat`（拍点）、`duration`（时值，拍）、`pitch`（MIDI）、`locked`（锁定） |
| `cylinder` | `diameter_mm` 滚筒直径、`effective_length_mm` 有效长度、`rpm` 转速 |
| `comb[]` | 每片簧片的 `pitch`（音高）与 `axial_mm`（轴向位置） |
| `constraints` | `pin_diameter_mm` 钉径、`seam_zone_mm` 接缝禁区弧宽、`min_rebound_seconds` 同簧片回弹间隔、`min_clearance_mm` 相邻钉最小净距 |
| `bpm` | 乐曲速度（拍/分） |

换算规则：

```
圆周 C = π · 直径
每拍秒数 spb = 60 / (bpm · tempo_factor)
角度 θ = (beat · spb · 6 · rpm) mod 360        （6·rpm = 每秒转过的度数）
周向坐标 x = θ / 360 · C                        （展开图横轴）
轴向坐标 y = 该音高簧片的 axial_mm              （展开图纵轴）
```

返回 `pins`（每枚钉的角度/坐标/余量）与 `diagnostics.first_by_kind` ——
**缺音、和弦碰撞、跨接缝、重复音过密、相邻钉净距**五类问题各自最早发生的位置：

```json
"first_by_kind": {
  "missing_pitch":   {"beat": 1.0, "note_ids": ["m"], ...},
  "chord_collision": {"beat": 4.0, "note_ids": ["c1", "c2"], ...},
  "seam_crossing":   {"beat": 0.0, "note_ids": ["s"], ...},
  "rebound_too_dense": {"beat": 6.0, "note_ids": ["r1", "r2"], ...},
  "pin_clearance":   {"beat": 10.0, "note_ids": ["q1", "q2"], ...}
}
```

冲突判定（在展开的圆柱面上计算，角度按圆周回绕取最短弧）：

- **缺音**：音符音高在音梳上没有对应簧片
- **和弦碰撞**：同时发声的钉，边缘净距 < `min_clearance_mm`
- **跨接缝**：钉边缘落入接缝线两侧 `seam_zone_mm/2` 范围内
- **重复音过密**：同一簧片相邻两次拨动的时间间隔 < `min_rebound_seconds`
- **相邻钉净距**：任意两钉边缘距离（轴向与周向弧长的合成）< `min_clearance_mm`

### 2. 搜索（search）

直接排不下时，在限定范围内搜索可制造方案：

```bash
curl -s -X POST localhost:8000/api/arrangements/search \
  -H 'Content-Type: application/json' -d '{
    "arrangement": { ...同上... },
    "limits": {
      "max_transpose_semitones": 3,     // 移调范围 ±3 半音
      "tempo_float_percent": 5.0,       // 速度浮动 ±5%
      "tempo_step_percent": 1.0,
      "quantize_grids_beats": [0.5, 0.25, 0.125],  // 量化网格（拍）
      "max_quantize_error_beats": 0.25,            // 量化误差上限
      "max_deletions": 30,
      "max_candidates": 5
    }
  }' | jq
```

- 枚举 移调 × 速度浮动 × 量化网格 组合；每种组合用贪心删减修复冲突
  （每步删除卷入冲突最多的未锁定音符），直到可制造或判定不可行
- 速度枚举围绕基准倍率 1.0 镜像对称：内点取在 ±k·步长 处，始终包含 1.0 与 ±浮动端点（步长不能整除浮动区间时同样对称）
- **锁定的音符（`locked: true`）不参与**移调、量化与删除；速度浮动是整机参数，对所有音符生效
- 候选指标（删减数、音高偏差、节奏误差、最小净距）只基于最终保留的音符计算，与冻结结果一致
- 候选按字典序排序：**删减音符数 → 音高偏差 → 节奏误差 → 最小净距（越大越好）**

### 3. 冻结（freeze）

把选定的方案冻结为不可变版本：

```bash
curl -s -X POST localhost:8000/api/versions \
  -H 'Content-Type: application/json' -d '{
    "arrangement": { ... },
    "solution": {
      "transpose_semitones": 12,
      "tempo_factor": 1.0,
      "quantize_grid_beats": null,
      "deleted_note_ids": []
    }
  }' | jq
```

- 冻结前服务端会重新计算并校验方案确实可制造（否则 422 并附诊断）
- `deleted_note_ids` 视为集合：重复项会被去重排序，不改变钉数、删减计数、哈希与版本
- **幂等**：相同 arrangement + solution 重复冻结返回同一版本（HTTP 200）
- 版本一旦创建不可修改、不可删除（没有更新/删除接口）

### 4. 读取版本

`GET /api/versions/{id}` 返回：

- 每枚钉的**音符来源**（`note_id`）、**坐标**（`angle_deg`、`x_mm`、`axial_mm`）、
  **冲突余量**（`margins`：与最近钉的净距余量、同簧片回弹余量、与接缝禁区余量）
- **输入哈希** `content_hash`：对规范化 JSON（arrangement + solution + pins + metrics）
  的 SHA-256；键排序、紧凑分隔符，保证跨进程稳定
- 可打印的 **SVG 展开图**：1 单位 = 1 mm，100% 比例打印即为真实尺寸钻孔模板；
  含接缝禁区（红）、簧片参考线与音高标注、钉位（绿=普通，蓝=锁定）、
  被删音符的红叉标记、角度标尺与方案参数标题栏
- `GET /api/versions/{id}/svg` 直接返回 `image/svg+xml`，可送浏览器打印

### 5. 重算校验（recompute）

```bash
curl -s -X POST localhost:8000/api/versions/1/recompute | jq
# {"id":1,"match":true,"stored_hash":"...","recomputed_hash":"..."}
```

从库中保存的原始请求重新计算，逐字段比对 pins/metrics/SVG 与哈希。
引擎是纯函数（无时钟、无随机、无环境依赖），同一版本重复计算必然得到相同结果。

## 动平衡

滚筒植钉后质量分布不再均匀，工作转速下会产生静不平衡与力偶不平衡。动平衡流程从
**已冻结的植钉版本**取数（滚筒几何 + 每枚钉的角度/轴向位置），机械参数独立保存：

| 字段 | 含义 |
| --- | --- |
| `shell_mass_g` | 滚筒壳体质量（克） |
| `pin_height_mm` | 钉高（质心半径 = 筒半径 + 钉高/2） |
| `default_pin_mass_g` / `pin_masses` | 默认钉质量与逐钉质量（键为来源版本的 note_id） |
| `working_rpm` | 工作转速（轴承载荷按 ω² 计算） |
| `bearing_a_mm` / `bearing_b_mm` | 两轴承轴向位置（可在筒外） |
| `plane_1_mm` / `plane_2_mm` | 两个校正面轴向位置（必须在筒长范围内） |
| `residual_limit_gmm` | 残余静不平衡上限（g·mm） |
| `inventory[]` | 配重库存：每种配重的 `mass_g`、`diameter_mm`、`quantity` |

### 1. 不平衡分析（analyze）

```bash
curl -s -X POST localhost:8000/api/balance/analyze \
  -H 'Content-Type: application/json' \
  -d @examples/balance_analyze_request.json | jq
```

把每枚钉换算为旋转质量矢量（质量 × 质心半径，按角度合成复矢量），返回：

- `imbalance`：质心偏移 `com_offset_mm`、静不平衡 `static_gmm`（及方向角）、
  相对中面的力偶 `couple_gmm2`、两轴承峰值载荷 `bearing_*_load_mn`（mN）、是否达标
- `pins[]`：逐钉贡献（静不平衡与力偶的 x/y 分量）
- `ideal_correction`：理论双面校正（各校正面所需质量与角度，供参考）
- 可附带 `locked_weights`（已装配配重），计入不平衡但不参与搜索

校验：来源版本不存在 → 404；逐钉质量引用未知钉号、校正面或配重越出筒长、
配重 id 未知、用量超库存、配重互相重叠 → 422；单位与几何范围由 Pydantic 核对。

### 2. 配重搜索（search）

```bash
curl -s -X POST localhost:8000/api/balance/search \
  -H 'Content-Type: application/json' \
  -d @examples/balance_search_request.json | jq
```

- 调用方可锁定已装配配重（`locked_weights`，消耗库存），并设置
  `angle_step_deg`（角度步长）、`pin_clearance_mm`（钉位安全距）、
  `seam_clearance_mm`（接缝安全距）、`max_weights`（最多配重数）
- 在库存余量内枚举配重位置：1–2 枚配重的方案穷举，更深方案由束搜索扩展；
  越出筒长、违反钉位/接缝安全距、与已装配配重干涉的位置一律排除
- 候选按字典序排序：**残余静不平衡 → 残余力偶 → 附加质量 → 配重数**

### 3. 冻结平衡方案（freeze）

```bash
curl -s -X POST localhost:8000/api/balance/plans \
  -H 'Content-Type: application/json' \
  -d @examples/balance_freeze_request.json | jq
```

- 冻结来源快照（版本哈希 + 滚筒几何 + 钉位）、机械参数、搜索限制与输入哈希；
  方案自包含，重算不依赖来源版本是否仍在库中
- **幂等**：相同输入重复冻结返回同一方案（HTTP 200）；方案不可修改、不可删除
- `GET /api/balance/plans/{id}/svg` 输出展开图：钉位（绿/蓝）、校正面（紫虚线 P1/P2）、
  轴承位置（灰点线 BA/BB）、中面线、已装配配重（灰）与新配重（橙，标注 id）

## 动力试算

植钉方案确定后，需要验证整曲走时：发条能否驱动滚筒转完所有钉、拨动瞬间掉速多少、
调速器能否把转速稳在设计值附近。试算版本**独立保存**整套动力参数并引用冻结的植钉版本：

| 字段 | 含义 |
| --- | ---|
| `spring_torque` | 发条扭矩曲线：扭矩（mN·m）对**剩余发条圈数**（turns，单调递增横坐标）分段线性 |
| `gear_train[]` | 齿轮级：`ratio` 为输出（滚筒）转速/输入（条盒）转速；总传动比为各级乘积 |
|  | `efficiency` 为该级效率（0,1]；总效率为各级乘积 |
| `spring_torque` 行程 | 横坐标跨度即可用发条行程（圈）；`prewind_turns` 不得超过它 |
| `base_inertia_g_cm2` | 滚筒轴上旋转组件惯量（g·cm²，1 g·cm² = 1e-7 kg·m²） |
| `governor_drag` | 调速器阻力矩曲线（mN·m）对滚筒转速（rpm，单调递增横坐标）分段线性 |
| `pluck_energy_uJ` | 音梳每枚簧片的**单次拨动耗能**（µJ），键为 MIDI 音高，须覆盖全部钉 |
| `dt_s` | 固定积分步长（秒，默认 1ms，范围 5e-5…0.02） |
| `stall_rpm_ratio` / `overspeed_rpm_ratio` / `beat_drift_limit` | 停转/超速/节拍漂移阈值 |

校验：来源版本不存在 → 404；曲线横坐标非严格递增、x/y 长度不等、单位不符、负值、
缺簧片耗能、**可用发条行程不足**（所需条盒圈数 = 滚筒转数/总传动比 > 曲线行程）→ 422。

### 1. 建立试算版本（trial）

```bash
curl -s -X POST localhost:8000/api/dynamics/trials \
  -H 'Content-Type: application/json' \
  -d @examples/dynamics_trial_request.json | jq
```

返回的 `derived` 给出设计转速、总传动比/效率、换算后的 SI 惯量、整曲所需滚筒转数与
条盒圈数、可用发条圈数；`pins[]` 是来源版本钉的设计相位（`phase_rev` 不回绕圈数）。
试算版本不可修改、不可删除；相同来源 + 参数重复创建返回同一版本（幂等）。

### 2. 仿真（simulate）

```bash
curl -s -X POST localhost:8000/api/dynamics/trials/1/simulate \
  -H 'Content-Type: application/json' \
  -d @examples/dynamics_simulate_request.json | jq
```

仿真以设计转速为初速度，按 `dt_s` 固定步长积分：

```
驱动扭矩（滚筒轴）= 发条扭矩(剩余圈数) / 总传动比 × 总效率
净扭矩            = 驱动扭矩 − 调速器系数 × 调速器阻力矩(转速)
越过钉的设计相位：ω_after = √(ω² − 2E拨动/J)     （同刻和弦合并为一次拨动）
剩余圈数          = 预紧圈数 − 滚筒累计转角/2π/总传动比
```

返回：

- `curves`：`t_s` / `rpm` / `torque_margin_mNm`（驱动−调速器的扭矩余量）等距采样曲线
- `pluck_events[]`：每次拨动的实际时刻、拨动前后转速、**到下一次拨动前的最低转速**、
  相对设计节拍间隔的漂移比、耗能；同刻多钉合并为一个事件
- `summary`：是否走完全程（`completed`）、**累计走时**、已用/剩余发条圈数、全程最低转速、
  最小扭矩余量、最大节拍漂移、违规计数
- `first_violations`：**首次停转、超速、节拍漂移的钉位（pin_index + note_ids）与时刻**；
  未发生为 null

### 3. 方案搜索（search）

```bash
curl -s -X POST localhost:8000/api/dynamics/trials/1/search \
  -H 'Content-Type: application/json' \
  -d @examples/dynamics_search_request.json | jq
```

- 调用方可 `gear_ratio_lock` **锁定传动比**；否则取 `gear_ratio_candidates`
  （缺省为存储的总传动比）
- 网格枚举 预紧圈数 × 调速器系数 × 飞轮惯量（附加到滚筒轴）；预紧超过可用行程的点跳过
- 排序字典序：**违规总数 → 最大节拍偏差 → 最小扭矩余量（越大越前）→
  剩余发条行程（越大越前）**，再用参数元组保证确定性
- 候选含完整指标、逐拨动最低转速序列与首次违规定位；`feasible` 表示存在零违规方案

### 4. 冻结选定方案（plan）

```bash
curl -s -X POST localhost:8000/api/dynamics/plans \
  -H 'Content-Type: application/json' \
  -d @examples/dynamics_plan_request.json | jq
```

- 冻结**来源快照（版本哈希 + 钉位相位）、输入曲线、积分步长 `dt_s`、解析后的场景与输入哈希**；
  方案自包含，重算不依赖试算或来源版本是否仍在库
- **幂等**：相同输入返回同一方案（HTTP 200）；不可修改、不可删除

### 5. 重算校验

```bash
curl -s -X POST localhost:8000/api/dynamics/plans/1/recompute | jq
# {"id":1,"match":true,"stored_hash":"...","recomputed_hash":"..."}
```

从库内快照与曲线重新积分，逐字段比对曲线/事件/汇总与哈希。引擎为纯函数，同一方案
重复计算（含跨进程）结果一致。

## 装配校准

植钉滚筒装上机架后，滚筒轴线相对音梳会有倾斜与偏心，钉与簧片的对位也会漂移。
校准批次从**已冻结的植钉版本**取数（滚筒几何 + 每枚钉的角度/轴向位置/音高），
把装配测量独立保存；批次状态在 **采集中 → 已确认** 之间流转（确认任一方案后即为
已确认，行本身不可变）。

机架坐标系：z 沿滚筒轴向，x 指向音梳。轴承高度定位旋转轴；径向跳动是壳面相对
旋转轴的有符号半径偏差（正 = 半径偏大），测量角度与植钉版本同参考（0° = 接缝）。

| 字段 | 含义 |
| --- | --- |
| `runout` | 径向跳动网格：`datum_points_mm` 基准点（≥2，严格递增）、`angles_deg` 测角（≥3，严格递增）、`values_mm` 每基准点一行读数 |
| `bearing_a_mm` / `bearing_b_mm` | 两轴承轴向位置（可在筒外） |
| `bearing_a_height_mm` / `bearing_b_height_mm` | 两轴承实测高度（旋转轴在机架中的位置） |
| `default_pin_height_mm` / `pin_heights_mm` | 默认钉高与逐钉钉高（键为来源版本的 note_id） |
| `reeds[]` | 每片簧片尖端：`axial_mm` 轴向位置、`height_mm` 尖端高度、`width_mm` 轴向宽度、`max_pluck_depth_mm` 允许拨入量 |
| `min_engagement_mm` | 发声所需最小拨入量 |
| `length_unit` / `angle_unit` | 单位声明，仅接受 `"mm"` / `"deg"` |

测点、单位和覆盖范围由 FastAPI/Pydantic 与引擎共同校验：基准点/测角严格递增、
网格行列对齐、簧片不重叠、最小拨入量小于每片簧片的允许拨入量（422）；基准点落在
筒长内且**覆盖全部钉的轴向范围**、每个钉音高都有实测簧片、逐钉钉高引用已知钉号（422）。

### 1. 建立校准批次（采集中）

```bash
curl -s -X POST localhost:8000/api/calibration/batches \
  -H 'Content-Type: application/json' \
  -d @examples/calibration_batch_request.json | jq
```

对每个基准点把跳动拟合为 r(θ) = a + b·cosθ + c·sinθ（(b,c) 即该处壳体偏心矢量，
a 为平均半径偏差），再沿轴向拟合直线得到**滚筒轴线与偏心**；逐钉解析返回：

- `contacted_pitches` **实际接触簧片**、`axial_deviation_mm` 轴向偏差、
  `pluck_depth_mm` 拨入量及 `axial_margin_mm` / `depth_margin_mm` 余量
- `diagnostics.first_by_kind`：**漏拨、错拨相邻簧片、同时擦碰两片、拨入过深**
  四类违规各自的首个钉位（pin_index + note_id）

```
钉尖高度 = 轴承插值的轴线高度(z) + 筒半径 + 拟合跳动(z, θ) + 钉高
拨入量   = 钉尖高度 − 簧片尖端高度
```

违规判定：与目标簧片无轴向重叠或拨入量不足 → 漏拨；只碰到相邻簧片 → 错拨；
同时重叠两片 → 擦碰两片；拨入量超过允许值 → 过深。相同来源 + 测量重复建批返回
同一批次（幂等）。

### 2. 调整方案搜索（search）

```bash
curl -s -X POST localhost:8000/api/calibration/batches/1/search \
  -H 'Content-Type: application/json' \
  -d @examples/calibration_search_request.json | jq
```

- 制作者可 `lock_bearing_a` / `lock_bearing_b` / `lock_comb_shift` / `lock_comb_height`
  **锁定不可改的轴承或音梳位置**（对应维度固定为 0）
- 枚举 两端轴承垫片（库存厚度、每端至多 `max_shims_per_end` 片的所有可达总厚）×
  音梳横移（±range 对称网格）× 高度调整（±range 对称网格）
- 候选按字典序排序：**违规数 → 最小余量（越大越前）→ 调整量 → 垫片种数**，
  再以调整元组保证确定性；同一总厚的垫片分解取 种数最少、片数最少 的代表

### 3. 确认（confirm）与重算

```bash
curl -s -X POST localhost:8000/api/calibration/batches/1/confirm \
  -H 'Content-Type: application/json' \
  -d @examples/calibration_confirm_request.json | jq
```

- 校验选定调整在包络内（锁定维度为 0、范围内、垫片总厚可达），否则 422
- 冻结**来源快照（版本哈希 + 滚筒几何 + 钉位）、全部测量、拟合参数、选定调整
  （垫片分解到库存厚度）与输入哈希**；方案自包含，重算不依赖来源版本是否在库
- **幂等**：相同调整 + 包络重复确认返回同一方案（HTTP 200）；批次随即转为已确认
- `POST /api/calibration/plans/{id}/recompute` 从库内快照重算并逐字段比对，
  结果逐字节一致

## 项目结构

```
musicbox/
  models.py              Pydantic 请求/响应模型与校验（植钉）
  engine.py              几何换算、冲突诊断、方案搜索（纯函数，确定性）
  balance_models.py      动平衡请求/响应模型与校验
  balance_engine.py      旋转质量矢量、不平衡计算、配重搜索（纯函数，确定性）
  balance_freeze.py      平衡方案的规范化哈希与冻结载荷构建
  dynamics_models.py     动力试算请求/响应模型与曲线、单位、参数范围校验
  dynamics_engine.py     固定步长滚筒角速度积分、拨动负载、方案搜索（纯函数，确定性）
  dynamics_freeze.py     试算版本与动力方案的规范化哈希与冻结载荷构建
  calibration_models.py  装配校准请求/响应模型与测点、单位、包络校验
  calibration_engine.py  滚筒轴线/偏心拟合、逐钉接触解析、调整搜索（纯函数，确定性）
  calibration_freeze.py  校准批次与确认方案的规范化哈希与冻结载荷构建
  svg.py                 滚筒展开图渲染（毫米单位，1:1 打印）
  freeze.py              规范化哈希与冻结载荷构建（植钉）
  store.py               SQLite 版本存储（只增不改：versions + balance_plans + dynamics_* + calibration_*）
  main.py                FastAPI 路由
tests/        pytest：引擎单元测试 + API 集成测试（138 项）
examples/     示例请求
```

## 备注

- 冲突检测为 O(n²) 钉对扫描，搜索复杂度为 组合数 × O(n²)；
  音乐盒曲目规模（数十至数百音符）在本机为亚秒至数秒级
- 搜索空间上限 5000 组合，超出返回 422，请收窄移调范围或速度步长
- 动平衡单位约定：质量 g、长度 mm、角度 度、转速 rpm；静不平衡 g·mm、
  力偶 g·mm²、轴承载荷 mN（峰值，载荷按 ω² 缩放，保留 9 位小数以免低载荷下失真）
- 配重搜索：1–2 枚配重穷举（标准双面校正为精确解），3 枚及以上由束搜索
  （宽度 32）扩展；目标值在 1e-9 处取整，消除浮点噪声对排序的干扰
- 动力试算单位约定：扭矩 mN·m、能量 µJ、惯量 g·cm²、转速 rpm，内部统一换算为 SI；
  曲线分段线性插值，端点外取端值；拨动耗能以动能冲量形式在越过钉相位的步内按线性
  插值时刻施加，和弦同刻合并。步长是被冻结的输入（默认 1ms）：换步长会改变数值结果，
  但同一步长重复积分逐字节一致
- 装配校准单位约定：长度 mm、角度 度；跳动拟合为闭式最小二乘（每基准点 3×3 简正
  方程 + 沿轴直线拟合），完全确定；调整搜索上限 20000 组合，超出返回 422，
  请收窄垫片规格、调整范围或步长
