"""分段运动模型影子诊断（开发用，不入镜像，不改正式输出）。

对每个候选帧比较三种模型，输出拟合误差与改善比，用于判断"曲率变化"能否
补充两点折角模型漏掉的触台（重点 v000028@f35）。

模型（以候选帧 f 为原点，t = frame - f，窗口 [f-w, f+w]）：
  B 连续直线  y = p*t + q                     (2 参数)
  C 触台分段  y = p1*t + q (t<=0) / p2*t + q (t>0)，位置在 f 连续 (3 参数)
  D 抛物线    y = a*t^2 + p*t + q             (3 参数)

比较：用 BIC（n*log(RSS/n) + k*log(n)，k=参数数）做复杂度惩罚；
改善比 G_BC = (BIC_B - BIC_C)/|BIC_B|、G_BD = (BIC_B - BIC_D)/|BIC_B|。

运行:  python shadow_models.py --traj <traj> --input <public_data>
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "participant" / "pingpang"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "participant" / "pingpang" / "participant"))
import numpy as np
from participant.landing import fuse_obs


def load_obs(traj, vid):
    pf = defaultdict(list)
    for l in Path(f"{traj}/traj_{vid}.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        pf[int(d["f"])].append((d["x"], d["y"], d["c"]))
    fused = fuse_obs(pf)
    return sorted((f, x, y, c) for f, (x, y, c) in fused.items())


def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, np.sum((A @ coef - y) ** 2)


def fit_kink(t, y):
    """触台分段：t<=0 一条线、t>0 一条线，共享截距 q（位置在 f 连续）。"""
    n = len(t)
    A = np.zeros((n, 3))
    for i, ti in enumerate(t):
        if ti <= 0:
            A[i, 0] = ti      # p1
        else:
            A[i, 1] = ti      # p2
        A[i, 2] = 1.0         # q
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, np.sum((A @ coef - y) ** 2)


def fit_parabola(t, y):
    A = np.vstack([t * t, t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, np.sum((A @ coef - y) ** 2)


def bic(rss, n, k):
    if n <= k:
        return float("inf")
    return n * math.log(rss / n + 1e-9) + k * math.log(n)


def scan(points, w=6):
    """对每个有足够邻点的帧，比较三模型，返回按帧的 BIC 与改善比。"""
    F = np.array([p[0] for p in points], float)
    Y = np.array([p[2] for p in points], float)
    rows = []
    for i, f in enumerate(F):
        idx = np.where((F >= f - w) & (F <= f + w))[0]
        if len(idx) < 5:
            continue
        t = F[idx] - f
        y = Y[idx]
        n = len(idx)
        _, rss_b = fit_line(t, y)
        _, rss_c = fit_kink(t, y)
        _, rss_d = fit_parabola(t, y)
        b_b, b_c, b_d = bic(rss_b, n, 2), bic(rss_c, n, 3), bic(rss_d, n, 3)
        rows.append((float(f), b_b, b_c, b_d,
                     (b_b - b_c) / abs(b_b) if b_b != 0 else 0.0,
                     (b_b - b_d) / abs(b_b) if b_b != 0 else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", default=None)
    a = ap.parse_args()

    from core.dataset import load_manifest, resolve_manifest
    videos = {v.video_id: v for v in load_manifest(resolve_manifest(a.input))}
    gt = defaultdict(list)
    for l in Path(a.gt or f"{a.input}/gt_public.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("valid"):
            gt[d["video_id"]].append((int(d["frame_id"]), float(d["x"]), float(d["y"])))

    # 已知 FN（无速度反号 / 投影滞后）
    focus = [("v000028", 35, "0度"), ("v000106", 7, "90度"), ("v000106", 158, "90度")]

    print("== 影子模型诊断：已知 FN 的 BIC 改善比（越大说明该模型越优于连续直线）==")
    for vid, gtf, ang in focus:
        pts = load_obs(a.traj, vid)
        rows = scan(pts, w=6)
        near = [r for r in rows if abs(r[0] - gtf) <= 3]
        print(f"\n{vid}@{gtf} [{ang}] GT 附近帧的 (f, G_BC=kink改善, G_BD=曲率改善)：")
        for r in sorted(near):
            print(f"  f={r[0]:.0f}  G_BC={r[4]:+.3f}  G_BD={r[5]:+.3f}")
        # 全局 kink / 曲率改善最大的帧
        best_kink = max(rows, key=lambda r: r[4])
        best_curv = max(rows, key=lambda r: r[5])
        print(f"  全局 G_BC 最大帧={best_kink[0]:.0f}(G={best_kink[4]:.3f})  "
              f"G_BD 最大帧={best_curv[0]:.0f}(G={best_curv[5]:.3f})  GT={gtf}")


if __name__ == "__main__":
    main()
