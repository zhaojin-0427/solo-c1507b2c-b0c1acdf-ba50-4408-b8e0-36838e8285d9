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
- 速度枚举关于基准倍率 1.0 对称，始终包含 1.0 与 ±浮动端点（步长不能整除浮动区间时也不会丢失基准值）
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

## 项目结构

```
musicbox/
  models.py   Pydantic 请求/响应模型与校验
  engine.py   几何换算、冲突诊断、方案搜索（纯函数，确定性）
  svg.py      滚筒展开图渲染（毫米单位，1:1 打印）
  freeze.py   规范化哈希与冻结载荷构建
  store.py    SQLite 版本存储（只增不改）
  main.py     FastAPI 路由
tests/        pytest：引擎单元测试 + API 集成测试（36 项）
examples/     示例请求
```

## 备注

- 冲突检测为 O(n²) 钉对扫描，搜索复杂度为 组合数 × O(n²)；
  音乐盒曲目规模（数十至数百音符）在本机为亚秒至数秒级
- 搜索空间上限 5000 组合，超出返回 422，请收窄移调范围或速度步长
