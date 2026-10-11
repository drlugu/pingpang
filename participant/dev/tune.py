#!/usr/bin/env python3
"""用缓存的逐帧观测在 CPU 上网格搜索落点参数（不需要 GPU/视频）。

步骤：
  1. 在平台/容器里跑一次推理并落盘观测（很便宜，只多写几个小 json）：
       PP_DUMP_DET=1 python3 run.py --input ... --output ...
  2. 在任意机器上（只需 numpy）调参：
       python3 participant/dev/tune.py --detections <out>/detections --write

``--write`` 会把每个视角的最优参数写回 ``participant/configs/landing.json``
（写完必须再跑一次推理，才能得到用新参数生成的最终 predictions.jsonl）。
各视角原有的 ``roi`` 等视角先验会被保留；``--calib-roi PX`` 可用真值落点包围盒
重新标定 ROI（见 ``roi_from_gt``）。

搜索空间里的重活只算一次：轨迹清洗按 (conf_gate, max_gap) 缓存，"折角"信号按
(conf_gate, max_gap, k, span) 缓存；阈值 / 方向比 / 强折角档 / NMS 搜索都是纯筛选，极快。
打分走 ``pick_landings`` + ``keep_in_roi``，与生产 ``Solution._analyze`` 同口径。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # pingpang/
sys.path.insert(0, str(ROOT))

from participant.dev.eval_f1 import match_counts    # noqa: E402
from participant.landing import (clean_track, fuse_obs,  # noqa: E402
                                 keep_in_roi, kink_candidates, pick_landings,
                                 table_roi)

ANGLES = ("0度", "45度", "90度")

CONF_GRID = (0.5, 0.6, 0.7, 0.8)
CLEAN_GAP_GRID = (4, 6)
K_GRID = (2, 3)
SPAN_GRID = (4.0, 6.0, 8.0)
MAXSTEP_GRID = (1.0, 2.0, 3.0, 1e9)
DV_GRID = (-8.0, -10.0, -12.0, -15.0)
# 90 度实测最优在 0.75~1.0 一带（1e9 = 关掉方向约束，反而因候选变多被 NMS 选错）
RATIO_GRID = (0.5, 0.35, 0.25, 0.75, 1.0, 1e9)
GAP_GRID = (5.0, 6.0, 7.0)
NMS_GRID = ("maxy", "dv", "combo")
# 强折角宽松档：|Δv_y| 够大时把方向比放宽到 ratio_strong（0 = 关闭该档）
DV_STRONG_GRID = (0.0, 20.0, 25.0, 30.0)
RATIO_STRONG_GRID = (0.6, 0.7, 0.8)


def f1_of(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load_gt(path: str | Path) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for line in Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.setdefault(r["video_id"], []).append(
            (int(r["frame_id"]), float(r["x"]), float(r["y"]),
             bool(r.get("valid", True))))
    return out


def load_dets(path: str | Path) -> list[dict]:
    docs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(path).glob("*.json"))]
    if not docs:
        raise SystemExit(f"{path} 下没有 *.json —— 请先用 PP_DUMP_DET=1 跑一次推理")
    return docs


def score_group(docs: list[dict], gt: dict, cands: dict[int, list], cfg: dict):
    """对一组视频（同一视角）按 cfg 打分，返回 (f1, tp, fp, fn)。

    cfg 里的 ``roi``（归一化球台投影区）会像生产代码那样做落点门控——
    调参口径必须与 ``Solution._analyze`` 完全一致，否则搜出来的"最优"不可复现。
    """
    tp = fp = fn = 0
    for d in docs:
        P = [(f, x, y) for f, x, y, _s in pick_landings(
            cands[d["video_id"]], cfg["dv_y"], cfg["max_dvx_ratio"],
            cfg["min_gap"], cfg["nms"], cfg.get("dv_strong", 0.0),
            cfg.get("ratio_strong", 0.0))]
        roi = table_roi(cfg.get("roi"), d["width"], d["height"])
        if roi:
            P = [(f, x, y) for f, x, y, _s in keep_in_roi(
                [(f, x, y, 0.0) for f, x, y in P], roi,
                min_keep=cfg.get("roi_min_keep", 0.5))]
        a, b, c, _ign = match_counts(P, gt.get(d["video_id"], []))
        tp += a
        fp += b
        fn += c
    return f1_of(tp, fp, fn), tp, fp, fn


def roi_from_gt(docs: list[dict], gt: dict, margin_px: float) -> list[float]:
    """用真值落点包围盒 + ``margin_px`` 外扩，标定该视角的球台投影区（归一化）。

    ``configs/landing.json`` 里各视角 ``roi`` 就是这么来的：ROI 需要"球台在画面中的
    位置"这个先验，而先验只能来自真值落点或人工标台——真值落点包围盒是最省事的一档。
    逐视频按各自分辨率归一化后取并集，因此换分辨率仍可用。
    """
    x0s, y0s, x1s, y1s = [], [], [], []
    for d in docs:
        pts = gt.get(d["video_id"], [])
        if not pts:
            continue
        w, h = float(d["width"]), float(d["height"])
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        x0s.append((min(xs) - margin_px) / w)
        x1s.append((max(xs) + margin_px) / w)
        y0s.append((min(ys) - margin_px) / h)
        y1s.append((max(ys) + margin_px) / h)
    if not x0s:
        return []
    return [round(max(0.0, min(x0s)), 4), round(max(0.0, min(y0s)), 4),
            round(min(1.0, max(x1s)), 4), round(min(1.0, max(y1s)), 4)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="乒乓球落点参数网格搜索")
    ap.add_argument("--detections", required=True)
    ap.add_argument("--gt", default="public_data/gt_public.jsonl")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                            / "configs" / "landing.json"))
    ap.add_argument("--write", action="store_true", help="把最优参数写回 config")
    ap.add_argument("--calib-roi", type=float, default=None, metavar="PX",
                    help="用真值落点包围盒 + PX 外扩标定各视角 roi（配合 --write 落盘）")
    ap.add_argument("--calib-roi-90", type=float, default=None, metavar="PX",
                    help="90 度的 roi 外扩（默认 = --calib-roi 的 2 倍：侧视把台面纵深"
                         "压得很窄，实测需要更大外扩）")
    a = ap.parse_args(argv)

    gt = load_gt(a.gt)
    docs = load_dets(a.detections)
    by_angle: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_angle[d["angle"]].append(d)
    print(f"载入 {len(docs)} 条视频：" +
          " ".join(f"{k}={len(v)}" for k, v in by_angle.items()))

    # ── 缓存：轨迹清洗（conf_gate/max_gap）→ 折角信号（k/span）────────────
    tracks: dict[tuple, dict] = {}
    cands: dict[tuple, dict] = {}
    for cg, clg in itertools.product(CONF_GRID, CLEAN_GAP_GRID):
        tr = {}
        for d in docs:
            obs = defaultdict(list)
            for f, x, y, c in d["points"]:
                obs[int(f)].append((x, y, c))
            pts = sorted((f, x, y, c) for f, (x, y, c) in fuse_obs(obs).items())
            g = [p for p in pts if p[3] >= cg]
            tr[d["video_id"]] = clean_track(g if len(g) >= 3 else pts,
                                            max_resid=30.0, max_gap=clg)
        tracks[(cg, clg)] = tr
        for k, span, mstep in itertools.product(K_GRID, SPAN_GRID, MAXSTEP_GRID):
            cands[(cg, clg, k, span, mstep)] = {
                vid: kink_candidates(pts, k, span, mstep) for vid, pts in tr.items()}
    print(f"缓存完成：{len(tracks)} 组轨迹 / {len(cands)} 组折角信号")

    base_cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    new_land = dict(base_cfg.get("landing", {}))

    # ── 全局粗搜：conf_gate / 清洗间隔 / k / span ──────────────────────────
    print("\n--- 全局（k/span/conf_gate/清洗间隔）粗搜 ---")
    glob = []
    for key, cc in cands.items():
        cg, clg, k, span, mstep = key
        tot = 0.0
        for ang in ANGLES:
            if by_angle[ang]:
                tot += score_group(by_angle[ang], gt, cc,
                                   {"dv_y": -10.0, "max_dvx_ratio": 0.5,
                                    "min_gap": 6.0, "nms": "maxy"})[0]
        glob.append((tot, key))
        print(f"k={k} span={span:>3} max_step={mstep:>4} conf_gate={cg} "
              f"清洗max_gap={clg}  粗F1={tot:.4f}")
    glob.sort(reverse=True)
    tot0, key = glob[0]
    cg, clg, k, span, mstep = key
    print(f"\n选定 k={k} span={span} max_step={mstep} conf_gate={cg} "
          f"清洗max_gap={clg}（粗F1={tot0:.4f}）\n")

    # ── 逐视角精搜：Δv_y / 方向比 / NMS 间隔 / NMS 策略 / 强折角宽松档 ─────────
    total = 0.0
    cc = cands[key]
    if a.calib_roi is not None:
        for ang in ANGLES:
            if not by_angle[ang]:
                continue
            if ang == "90度":
                margin = (a.calib_roi_90 if a.calib_roi_90 is not None
                          else 2.0 * a.calib_roi)
            else:
                margin = a.calib_roi
            roi = roi_from_gt(by_angle[ang], gt, margin)
            if roi:
                new_land.setdefault(ang, {})["roi"] = roi
                print(f"{ang}: ROI(外扩{margin:g}px) = {roi}")
    for ang in ANGLES:
        if not by_angle[ang]:
            continue
        own = {**base_cfg.get("landing", {}).get(ang, {}),
               **new_land.get(ang, {})}      # 保留 ROI 等已有字段（含刚标定的）
        best = None
        for dv, ratio, gap, nms, dvs, rs in itertools.product(
                DV_GRID, RATIO_GRID, GAP_GRID, NMS_GRID,
                DV_STRONG_GRID, RATIO_STRONG_GRID):
            cfg = {"dv_y": dv, "max_dvx_ratio": ratio, "min_gap": gap, "nms": nms,
                   "dv_strong": dvs, "ratio_strong": rs}
            f1, tp, fp, fn = score_group(by_angle[ang], gt, cc, {**own, **cfg})
            if best is None or f1 > best[0]:
                best = (f1, dict(cfg), tp, fp, fn)
        f1, cfg, tp, fp, fn = best
        total += f1
        new_land[ang] = {**own, **cfg, "k": k, "span": span, "max_step": mstep,
                         "max_resid": 30.0, "max_gap": clg}
        print(f"{ang}: F1={f1:.4f}  TP={tp} FP={fp} FN={fn}  {cfg}")

    print(f"\n最优参数下精度分 = {total * 25:.2f} / 75")

    if a.write:
        base_cfg.setdefault("detector", {})["conf_gate"] = cg
        base_cfg["landing"] = new_land
        # default 档＝复制 45 度档（与历史行为一致），但 roi / dv_strong / ratio_strong
        # 是视角相关的先验，不往 default 里塞（未知视角宁可不过滤、不放宽）。
        fallback = {k: v for k, v in new_land.get("45度", {}).items()
                    if k not in ("roi", "dv_strong", "ratio_strong")}
        base_cfg["landing"]["default"] = {**new_land.get("default", {}), **fallback}
        Path(a.config).write_text(
            json.dumps(base_cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"已写入 {a.config}（记得重新跑一次推理生成最终 predictions.jsonl）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
