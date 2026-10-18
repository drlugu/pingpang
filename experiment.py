"""实验回归工具（开发用，不入镜像）：对 TRT 轨迹 dump 重放完整管线并评分。

用法:
  python experiment.py --traj <traj_dir> [--override '{"45度":{"max_resid":30}}'] [--detail]

对比基线（无 override）与实验（带 override）的逐视角 TP/FP/FN 与逐视频变化。
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


def match_counts(P, G):
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


def run_all(videos, cfg, traj, detail=False):
    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for v in VIEWS}
    per_vid = {}
    for vid, meta in videos.items():
        pts = load_obs(traj, vid)
        det = cfg["detector"]
        cg = float(det.get("conf_gate", 0.6))
        lc = S._deep_merge(cfg["landing"].get("default", {}), cfg["landing"].get(meta.angle, {}))
        preds = analyze_landings(pts, meta.angle, meta.width, meta.height, meta.n_frames, cg, lc)
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + [(f, x, y, False) for f, x, y in g["invalid"]]
        tp, fp, fn, _ = match_counts(preds, G)
        a = acc[meta.angle]
        a["tp"] += tp
        a["fp"] += fp
        a["fn"] += fn
        a["n"] += len(g["valid"])
        per_vid[vid] = (tp, fp, fn)
    score = 0.0
    for v in VIEWS:
        a = acc[v]
        p = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0
        r = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        a["f1"] = f1
        score += 25 * f1
    acc["score"] = score
    return acc, per_vid


def report(acc, per_vid, baseline_per_vid=None, title=""):
    if title:
        print(title)
    for v in VIEWS:
        a = acc[v]
        print(f"  {v}: TP={a['tp']:2d} FP={a['fp']} FN={a['fn']} GT={a['n']:2d} F1={a['f1']:.3f}")
    print(f"  score = {acc['score']:.2f}/75")
    if baseline_per_vid is not None:
        diffs = []
        for vid, (tp, fp, fn) in per_vid.items():
            bt, bf, bn = baseline_per_vid[vid]
            if (tp, fp, fn) != (bt, bf, bn):
                diffs.append(f"{vid}: TP {bt}->{tp} FP {bf}->{fp} FN {bn}->{fn}")
        print(f"  变化: {diffs if diffs else '无'}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--override", default=None, help='JSON, 如 {"45度":{"max_resid":30}}')
    ap.add_argument("--gt", default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.input).resolve().parent.parent))
    from core.dataset import load_manifest, resolve_manifest
    videos = {v.video_id: v for v in load_manifest(resolve_manifest(a.input))}
    global gt
    gt = load_gt(Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl")

    base_cfg = S.load_config()
    acc_b, per_b = run_all(videos, base_cfg, a.traj)
    report(acc_b, per_b, None, "== 基线 ==")

    if a.override:
        ov = json.loads(a.override)
        exp_cfg = S._deep_merge(base_cfg, {"landing": ov})
        acc_e, per_e = run_all(videos, exp_cfg, a.traj)
        report(acc_e, per_e, per_b, f"== 实验 override={a.override} ==")


if __name__ == "__main__":
    main()
