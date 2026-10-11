#!/usr/bin/env python3
"""landing.py 离线冒烟测试（只需 numpy，不需要模型/视频/GPU）。

    python3 participant/dev/test_landing.py

构造一段"分段抛物线 + 触台拐点"的仿真轨迹（真实乒乓球在两跳之间的 y(t)
是一条开口向上的抛物线，触台瞬间速度反向 → 拐点/局部极大），再叠加噪声与
离群点，验证 fuse_obs / clean_track / find_landings 能否把触台帧找回来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from participant.landing import clean_track, find_landings, fuse_obs  # noqa: E402

W, H = 1280, 720
Y_B = 600.0          # 台面在画面中的 y
G = 1.6              # 重力（px/帧²，图像坐标向下为正）
BOUNCES = (25, 60, 95)          # 真实触台帧


def sim_track(bounces=BOUNCES, n_frames=131, seed=0, noise=1.5):
    """分段抛物线轨迹；触台帧是 y 的局部极大（拐点）。

    每一跳从上一个触台帧到下一个触台帧，飞行时长 T 决定初速度 v0=g·T/2，
    这样两个端点正好都落在 y=Y_B（球台）上，顶点在中间。
    """
    rng = np.random.default_rng(seed)
    edges = [-9] + list(bounces) + [n_frames - 1]   # -9：首跳的"上一跳"在画面外
    y = np.zeros(n_frames, dtype=np.float64)
    for k in range(len(edges) - 1):
        s, e = edges[k], edges[k + 1]
        v0 = G * (e - s) / 2.0
        lo, hi = max(s, 0), min(e, n_frames - 1)
        t = np.arange(lo, hi + 1, dtype=np.float64)   # 闭区间，帧号全覆盖
        y[lo:hi + 1] = Y_B - v0 * (e - t) + 0.5 * G * (e - t) ** 2
    f = np.arange(n_frames, dtype=np.float64)
    x = 120.0 + 1.8 * f + 25.0 * np.sin(f / 35.0)
    conf = 0.85 + 0.1 * rng.random(n_frames)     # 检测置信度（模拟热图峰值）
    return [(int(i), float(x[i] + rng.normal(0, 0.5)),
             float(y[i] + rng.normal(0, noise)), float(conf[i]))
            for i in range(n_frames)]


def main() -> int:
    ok = True

    # ── 1. fuse_obs：同帧的多通道冗余观测应合并成一条 ─────────────────────
    fused = fuse_obs({10: [(100.0, 200.0, 0.9), (102.0, 201.0, 0.8)],
                      11: [(103.0, 202.0, 0.95)],
                      12: [(400.0, 400.0, 0.7), (104.0, 203.0, 0.9)]})
    assert set(fused) == {10, 11, 12}
    assert fused[11] == (103.0, 202.0, 0.95)
    avg = abs(fused[10][0] - 101.0) < 1e-6 and abs(fused[10][1] - 200.5) < 1e-6
    far = abs(fused[12][0] - 104.0) < 1e-6      # 远处那条被聚类过滤掉
    print(f"[1] fuse_obs  同帧两观测取均值={avg}  离群观测被滤除={far}")
    ok &= avg and far

    # ── 2. clean_track：注入离群点应被剔除，正常点应被保留 ─────────────────
    track = sim_track()
    dirty = list(track)
    dirty[70] = (70, track[70][1] + 90.0, track[70][2] - 120.0, 0.5)   # 离群
    cleaned = clean_track(dirty, max_resid=30.0, max_gap=6)
    lost = len(dirty) - len(cleaned)
    kept_outlier = any(p[0] == 70 for p in cleaned)
    print(f"[2] clean_track  剔除 {lost} 个点  离群点残留={kept_outlier}")
    ok &= (lost >= 1 and not kept_outlier)

    # ── 3. find_landings：触台帧/坐标是否命中（折角判据）──────────────────
    cfg = {"k": 2, "span": 6, "dv_y": -10.0, "max_dvx_ratio": 0.5,
           "min_gap": 6, "nms": "maxy"}
    det = find_landings(cleaned, cfg)
    for f, x, y, s in det:
        print(f"    检出 frame={f:3d} x={x:7.1f} y={y:6.1f} score={s:.2f}")
    n_ok = len(det) == len(BOUNCES)
    dframe, dcoord = [], []
    for b in BOUNCES:
        if det:
            near = min(det, key=lambda d: abs(d[0] - b))
            dframe.append(abs(near[0] - b))
            dcoord.append(abs(near[2] - Y_B))
        else:
            dframe.append(99)
            dcoord.append(99.0)
    hit = sum(1 for d in dframe if isinstance(d, int) and d <= 3)
    print(f"[3] find_landings  检出 {len(det)}/{len(BOUNCES)} 个落点，"
          f"帧偏差 {dframe}，坐标偏差 {[round(v, 1) for v in dcoord]}")
    print(f"    命中(帧偏差<=3) {hit}/{len(BOUNCES)}")
    ok &= n_ok and hit == len(BOUNCES)

    # ── 4. 输出合法性（对齐 validate.py 的规则）────────────────────────────
    legit = all(0 <= f < len(track) and 0 <= x < W and 0 <= y < H
                for f, x, y, _s in det)
    print(f"[4] 输出合法性（frame/x/y 均在合法区间）={legit}")
    ok &= legit

    # ── 5. 断口鲁棒性：飞行段人为挖出 4 帧缺口，不应冒出伪落点 ─────────────
    holed = [p for p in cleaned if p[0] not in (72, 73, 74)]
    det2 = find_landings(holed, cfg)
    ghost = [f for f, *_ in det2 if 68 <= f <= 80]
    real = sorted(f for f, *_ in det2)
    print(f"[5] 断口鲁棒性  挖空 72-74 帧后检出 {real}，缺口附近伪落点={ghost}")
    ok &= (not ghost and len(det2) == len(BOUNCES))

    print("\n" + ("PASS ✅  算法链路（融合→清洗→速度折角触台判定）行为符合预期"
                  if ok else "FAIL ❌  见上面细节"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
