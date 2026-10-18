"""已知错误样本的回归记录工具（开发用，不入镜像）。

固化 6 FN + 1 FP 的逐例状态，每次改动后运行，输出：
  1) 全量逐视角 TP/FP/FN + 精度分；
  2) 每个已知 FN 是否仍漏检（及最近预测的帧号/距离）；
  3) 已知 FP 是否仍误报。

运行:  python regression.py --traj <traj_dir> --input <public_data>
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "participant" / "pingpang"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "participant" / "pingpang" / "participant"))

from participant import solution as S
from participant.landing import analyze_landings, fuse_obs

VIEWS = ("0度", "45度", "90度")

# 已知错误样本（GT 的 video_id + frame_id）；FP 单独列
KNOWN_FN = [
    ("v000028", 35, "0度"),
    ("v000066", 83, "45度"),
    ("v000094", 43, "45度"),
    ("v000094", 79, "45度"),
    ("v000106", 7, "90度"),
    ("v000106", 158, "90度"),
]
KNOWN_FP = [
    ("v000038", 49, "45度"),
]


def load_obs(traj, vid):
    pf = defaultdict(list)
    for l in Path(f"{traj}/traj_{vid}.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        pf[int(d["f"])].append((d["x"], d["y"], d["c"]))
    fused = fuse_obs(pf)
    return sorted((f, x, y, c) for f, (x, y, c) in fused.items())


def load_gt(path):
    gt = defaultdict(lambda: {"valid": [], "invalid": []})
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        k = "valid" if d.get("valid") else "invalid"
        gt[d["video_id"]][k].append((int(d["frame_id"]), float(d["x"]), float(d["y"])))
    return gt


def match(P, G):
    adj = [[] for _ in P]
    for i, (pf, px, py) in enumerate(P):
        for j, (gf, gx, gy, _v) in enumerate(G):
            if abs(pf - gf) <= 1 and math.hypot(px - gx, py - gy) <= 50:
                adj[i].append(j)
    mr = [-1] * len(G)

    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if mr[v] == -1 or dfs(mr[v], seen):
                mr[v] = u
                return True
        return False

    for u in range(len(P)):
        dfs(u, [False] * len(G))
    matched = {u for u in mr if u != -1}
    tp = sum(1 for j, u in enumerate(mr) if u != -1 and G[j][3])
    fn = sum(1 for g in G if g[3]) - tp
    fp = len(P) - len(matched)
    return tp, fp, fn, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--gt", default=None)
    a = ap.parse_args()

    from core.dataset import load_manifest, resolve_manifest
    videos = {v.video_id: v for v in load_manifest(resolve_manifest(a.input))}
    gt = load_gt(Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl")
    cfg = S.load_config()

    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for v in VIEWS}
    per_vid_preds = {}
    for vid, meta in videos.items():
        pts = load_obs(a.traj, vid)
        cg = float(cfg["detector"].get("conf_gate", 0.6))
        lc = S._deep_merge(cfg["landing"].get("default", {}), cfg["landing"].get(meta.angle, {}))
        preds = analyze_landings(pts, meta.angle, meta.width, meta.height, meta.n_frames, cg, lc)
        per_vid_preds[vid] = preds
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + [(f, x, y, False) for f, x, y in g["invalid"]]
        tp, fp, fn, _ = match(preds, G)
        acc[meta.angle]["tp"] += tp
        acc[meta.angle]["fp"] += fp
        acc[meta.angle]["fn"] += fn
        acc[meta.angle]["n"] += len(g["valid"])

    score = 0.0
    print("== 全量 ==")
    for v in VIEWS:
        a = acc[v]
        p = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0
        r = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        score += 25 * f1
        print(f"  {v}: TP={a['tp']:2d} FP={a['fp']} FN={a['fn']} GT={a['n']:2d} F1={f1:.3f}")
    print(f"  精度分 = {score:.2f}/75")

    print("== 已知 FN 逐例 ==")
    for vid, gtf, ang in KNOWN_FN:
        g = gt.get(vid, {"valid": [], "invalid": []})
        # 找该 GT 的坐标
        gt_pt = None
        for f, x, y in g["valid"]:
            if f == gtf:
                gt_pt = (x, y)
                break
        preds = per_vid_preds.get(vid, [])
        # 是否命中（±1 帧且 ≤50px）
        hit = any(abs(p[0] - gtf) <= 1 and math.hypot(p[1] - gt_pt[0], p[2] - gt_pt[1]) <= 50
                  for p in preds)
        # 最近预测
        nearest = min(preds, key=lambda p: abs(p[0] - gtf)) if preds else None
        nd = (abs(nearest[0] - gtf),
              round(math.hypot(nearest[1] - gt_pt[0], nearest[2] - gt_pt[1]), 1)) if nearest else None
        status = "TP" if hit else "FN"
        print(f"  {vid}@{gtf} [{ang}]: {status} 最近预测帧={nearest[0] if nearest else None} "
              f"(帧差={nd[0]}, 距离={nd[1]})" if nd else f"  {vid}@{gtf} [{ang}]: {status} 无预测")

    print("== 已知 FP 逐例 ==")
    for vid, pf, ang in KNOWN_FP:
        preds = per_vid_preds.get(vid, [])
        near = [p for p in preds if abs(p[0] - pf) <= 1]
        # 该预测附近是否有 valid GT
        g = gt.get(vid, {"valid": [], "invalid": []})
        hit_gt = any(abs(f - pf) <= 1 and math.hypot(x - p[1], y - p[2]) <= 50
                     for f, x, y in g["valid"] for p in near)
        status = "FP(仍误报)" if near and not hit_gt else ("已消除/命中GT" if near else "已消除")
        print(f"  {vid}@{pf} [{ang}]: {status} 预测数={len(near)}")


if __name__ == "__main__":
    main()
