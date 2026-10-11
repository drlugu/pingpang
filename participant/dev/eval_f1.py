#!/usr/bin/env python3
"""本地评分：predictions.jsonl vs GT，按官方口径复刻。

    python3 participant/dev/eval_f1.py \
        --predictions /participant/output/pingpang/predictions.jsonl \
        --gt public_data/gt_public.jsonl \
        --manifest public_data/manifest.json

口径（与赛题一致）：
  * 帧号容差 ±1、位置容差 50 px，同一视频内一对一匹配；
  * 只有 ``valid=true`` 的 GT 参与召回统计（算 FN / 可被 TP）；
  * 命中 ``valid=false`` 干扰点（二次弹跳/擦边/遮挡）的预测被"忽略"，
    既不计 TP 也不计 FP；
  * TP 用最大基数二分匹配（Kuhn）求解，保证容差内命中数最大。

输出每个视角的 P / R / F1 与合成精度分 ``F1_0×25 + F1_45×25 + F1_90×25``。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

FRAME_TOL = 1
DIST_TOL = 50.0
ANGLES = ("0度", "45度", "90度")


def max_matching(adj: list[list[int]], n_right: int) -> list[int]:
    """Kuhn 最大基数匹配；返回长度 n_right 的 match（右侧点 -> 左侧点），-1 表示未匹配。"""
    match_r = [-1] * n_right

    def dfs(u: int, seen: list[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] == -1 or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    for u in range(len(adj)):
        dfs(u, [False] * n_right)
    return match_r


def match_counts(P: list[tuple], G: list[tuple]) -> tuple[int, int, int, int]:
    """单视频匹配，返回 (tp, fp, fn, ignored)。

    ``P`` = [(fid,x,y)]，``G`` = [(fid,x,y,valid)]。``ignored`` 是命中
    ``valid=false`` 干扰点、因此不计 TP 也不计 FP 的预测数。
    """
    adj = [[] for _ in P]
    for i, (pf, px, py) in enumerate(P):
        for j, (gf, gx, gy, _v) in enumerate(G):
            if abs(pf - gf) <= FRAME_TOL and math.hypot(px - gx, py - gy) <= DIST_TOL:
                adj[i].append(j)
    match_r = max_matching(adj, len(G)) if G else []
    matched = {u for u in match_r if u != -1}
    tp = sum(1 for j, u in enumerate(match_r) if u != -1 and G[j][3])
    ign = sum(1 for j, u in enumerate(match_r) if u != -1 and not G[j][3])
    fn = sum(1 for g in G if g[3]) - tp
    fp = len(P) - len(matched)
    return tp, fp, fn, ign


def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="乒乓球落点本地 F1 评测")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gt", default="public_data/gt_public.jsonl")
    ap.add_argument("--manifest", default="public_data/manifest.json")
    ap.add_argument("--show-misses", action="store_true", help="打印每个漏检/误检")
    a = ap.parse_args(argv)

    sys.setrecursionlimit(100000)

    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    vid2angle = {v["video_id"]: v["angle"] for v in man["videos"]}
    nframes = {v["video_id"]: v["n_frames"] for v in man["videos"]}

    preds = collections.defaultdict(list)
    unknown = 0
    for r in load_jsonl(a.predictions):
        vid = r.get("video_id")
        if vid not in vid2angle:
            unknown += 1
            continue
        preds[vid].append((int(r["frame_id"]), float(r["x"]), float(r["y"])))

    gts = collections.defaultdict(list)
    for r in load_jsonl(a.gt):
        gts[r["video_id"]].append((int(r["frame_id"]), float(r["x"]),
                                   float(r["y"]), bool(r.get("valid", True))))

    stat = {ang: {"tp": 0, "fp": 0, "fn": 0, "ign": 0} for ang in ANGLES}
    for vid, ang in vid2angle.items():
        if ang not in stat:
            continue
        P = preds.get(vid, [])
        G = gts.get(vid, [])
        tp, fp, fn, ign = match_counts(P, G)
        s = stat[ang]
        s["tp"] += tp
        s["fp"] += fp
        s["fn"] += fn
        s["ign"] += ign
        if a.show_misses and (fn or fp):
            print(f"  [{vid} {ang}] n={nframes[vid]} P={len(P)} TP={tp} "
                  f"FP={fp} FN={fn} ignored={ign}")

    print(f"\n{'视角':<6}{'TP':>5}{'FP':>5}{'FN':>5}{'忽略':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    score = 0.0
    for ang in ANGLES:
        s = stat[ang]
        p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        score += f1 * 25
        print(f"{ang:<6}{s['tp']:>5}{s['fp']:>5}{s['fn']:>5}{s['ign']:>6}"
              f"{p:>8.3f}{r:>8.3f}{f1:>8.3f}")
    print(f"\n精度分 = {score:.2f} / 75"
          + (f"   (未知 video_id 的预测 {unknown} 条，已忽略)" if unknown else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
