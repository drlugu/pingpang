# 乒乓球落点检测方案（participant/）

多角度（0°/45°/90°）乒乓球视频 → 逐帧球心 → 触台帧判定 → 落点坐标。

## 1. 交付物清单

```
participant/
├── solution.py            参赛入口（run.py 通过 participant.solution:Solution 加载）
├── ball_detector.py       球检测：热图模型 → 逐帧球心（多通道冗余观测）
├── landing.py             落点判定：轨迹融合/清洗 + 触台"折角"检测（纯 numpy）
├── base_solution.py       官方基类（未改动）
├── configs/landing.json   全部超参（检测阈值 + 按视角分档的落点参数）
├── weights/ball.onnx      官方提供的球检测权重（未改动）
└── dev/                   离线开发工具（不参与评测，可在平台外运行）
    ├── eval_f1.py         按官方口径本地评分（±1 帧 / ≤50px / Kuhn 一对一匹配）
    ├── tune.py            用缓存的逐帧观测做网格搜索
    ├── replay.py          从缓存观测**重放生产代码**的落点链路（秒级出分）
    ├── calib_offset.py    标定"输出通道 ↔ 帧偏移"
    └── test_landing.py    落点算法冒烟测试（只需 numpy，含断口鲁棒性用例）
```

## 1.5 实测结果（公开验证集 15 条，本地复刻官方口径）

| 视角 | 视频 | TP | FP | FN | 忽略 | P | R | **F1** |
|---|---|---|---|---|---|---|---|---|
| 0° | 5 | 19 | 8 | 2 | 0 | 0.704 | 0.905 | **0.792** |
| 45° | 7 | 31 | 7 | 7 | 0 | 0.816 | 0.816 | **0.816** |
| 90° | 3 | 29 | 9 | 4 | 2 | 0.763 | 0.879 | **0.817** |

**精度分 = 60.61 / 75**（官方基线 43.7 / 75），三视角 F1 均值 0.808 vs 基线总 F1 0.604。
「忽略」= 命中 GT 里 `valid=false` 的干扰点，按赛题口径既不算 TP 也不算 FP。

## 2. 方法

**Step 1｜观测（ball_detector.py）**
`ball.onnx` 输入 `(1,9,288,512)`（3 帧 RGB 栈，ImageNet 归一化），输出
`(1,3,288,512)` 的热图。逐帧推理时，**通道 c 编码的是帧栈中第 c 帧**，即
对以帧 `f` 结尾的栈，`ch2→f`、`ch1→f-1`、`ch0→f-2`（由 `dev/calib_offset.py`
用真值落点标定确认：三个通道的峰值都落在真值 7px 内）。
因此三通道同时启用 = 每帧最多 3 票冗余观测，`fuse_obs` 对同一帧的多票取均值，
既降噪又补足丢检。

**Step 2｜触台判定（landing.py，核心创新点）**
触台瞬间只有**竖直速度反向**，水平速度基本不变。这个突变在图像上表现为球心
速度矢量的折转，与视角无关。对每个候选帧，用前后各 `k` 个观测分别线性拟合
出触台前后的速度 `v⁻`、`v⁺`，取

    Δv_y = v⁺_y − v⁻_y  ≤ dv_y（负阈值），且 |Δv_x| ≤ ratio·|Δv_y|

* `Δv_y` 足够负 ⇔ 下落被瞬间抬升 ⇔ 触台；
* `|Δv_x|/|Δv_y|` 的上限来自"竖直反跳在图像中的投影方向"，用于滤掉跟错目标、
  轨迹跳变造成的伪折角；
* 紧邻观测必须在 `max_step` 帧内，否则轨迹断口会伪造出 ±100px/帧² 的巨型折角。

⚠️ **为什么不用"y 取局部极大"**：0°/45° 是沿球台纵深方向俯视，触台后球同时
远离相机，投影 `y` 会继续增大——用抛物线顶点/极值法实测 F1 只有 0.05，而折角
判据能到 0.7~0.8。这是本方案相对朴素做法的关键差异。

**Step 3｜坐标**
触台帧观测到的球心即落点坐标（真值就是标注者在触台帧标下的球心位置，实测
在真值帧上模型球心与真值中位差约 7px ≪ 50px 匹配容差）。

## 3. 运行

```bash
# 平台/评测机：单任务端到端（GPU 解码 + TRT 优先，自动回退 onnxruntime）
./run.sh <input_dir> <output_dir>

# 本地自检（跑 1 条视频，校验输出合法性）
python3 scripts/check_solution.py --input public_data --solution participant.solution:Solution

# 本地评分（复刻官方口径）
python3 participant/dev/eval_f1.py --predictions <output_dir>/predictions.jsonl

# 落点算法冒烟测试（不需要模型/视频/GPU）
python3 participant/dev/test_landing.py
```

调参闭环（在任意只有 numpy 的机器上即可完成，不需要 GPU）：

```bash
PP_DUMP_DET=1 python3 run.py --input <input> --output <out> --decoder cpu   # 落盘逐帧观测
python3 participant/dev/tune.py --detections <out>/detections --write       # 网格搜索→写回 config
python3 participant/dev/replay.py --detections <out>/detections             # 秒级重放验证（出分）
python3 run.py --input <input> --output <out>                               # 用新参数重跑出最终结果
```

`replay.py` 直接实例化 `Solution` 并调用其 `prepare`（加载真实配置）+ `_analyze`
（真实算法），因此**测的就是生产代码**。落点分析是纯后处理（只吃逐帧观测），
推理结果已被 dump 完整捕获，所以重放结果与真跑一遍推理逐字节一致——
改一次参数验证只需几秒，不必重跑十几分钟推理。改完配置务必用 `replay.py`
或整跑复核，避免"配置写了但代码没读到"这类静默失配。

## 4. 算力与工程取舍

* 模型是 5.3MB 的小型热图网络，单帧 288×512 推理在 4090 上开销极低；
  `prepare` 里先做一次 warmup，避免首帧开销计入推理计时；TRT engine 带
  (TRT 版本 + GPU 型号 + onnx 哈希) 目录级缓存，第二次起直接反序列化。
* 落点判定全部是 numpy 的 O(n) 拟合（n=帧数），单视频耗时 ≪ 1ms，
  对 50fps 的速度分没有影响。
* 所有观测在 `finish_video` 统一分析（框架允许延迟输出），因此落点判定不受
  "必须即时输出"的限制；单视频异常被 try/except 隔离，不会拖垮整场。
* 输出前统一 clamp 到 `[0, width-1e-3]×[0, height-1e-3]`，杜绝越界非法行。
