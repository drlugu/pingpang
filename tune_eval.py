"""离线精度评测 + 调参工具（开发用，**不属于提交产物**）。

本文件在仓库根，刻意不放进 ``participant/``——它会读 ``gt_public.jsonl`` 真值，
而 participant/ 整个目录会被打进提交镜像。

口径（与赛题细则一致）：
  * 帧号容差 ±1、位置容差 50px，同一视频内**一对一**匹配（Kuhn 最大基数匹配，
    保证容差内命中数最大——这是官方口径，贪心会低估 TP）；
  * 只有 ``valid=true`` 的 GT 参与召回统计（算 FN）；
  * 命中 ``valid=false`` 干扰点（二次弹跳/擦边/遮挡）的预测被"忽略"，
    既不计 TP 也不计 FP；
  * 精度分 = F1_0×25 + F1_45×25 + F1_90×25。

子命令：
  1) dump —— 跑一遍模型（复用提交的 BallDetector 多通道解码），把每帧的冗余
     观测落盘成 ``traj_<vid>.jsonl``（每行一个观测：f/x/y/c/offset）。
  2) score —— 纯离线：重放 landing.find_landings + 官方口径评分 + 参数扫描。
  3) selftest —— 不需要模型：验证评分器与重放链路。

设计原则：落点判定只有一份实现（participant/pingpang/participant/landing.py）。
本工具 import 它，所以「扫出来的参数」与「线上真正跑的判定」不可能对不上。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_task_root(start: Path) -> Path:
    """定位任务根 = 同时含 core/ 与 participant/ 的目录（.../participant/pingpang）。"""
    p = start.resolve()
    for cand in (p, *p.parents):
        for rel in (".", "pingpang", "participant/pingpang", "participant"):
            t = cand / rel if rel != "." else cand
            if (t / "core").is_dir() and (t / "participant").is_dir():
                return t
    return p.parent


ROOT = _find_task_root(_HERE)
for _p in (str(ROOT / "participant"), str(ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from participant.landing import analyze_landings, fuse_obs  # noqa: E402
from participant.ball_detector import BallDetector, IN_H, IN_W          # noqa: E402
from participant import solution as _solution                            # noqa: E402

HIT_FRAMES = 1
HIT_DIST = 50.0
VIEWS = ("0度", "45度", "90度")

# ── 数据加载 ────────────────────────────────────────────────────────────────
def load_gt(path: Path) -> dict:
    """→ {video_id: {"valid": [(f,x,y)], "invalid": [(f,x,y)]}}"""
    out = defaultdict(lambda: {"valid": [], "invalid": []})
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        key = "valid" if d.get("valid") else "invalid"
        out[d["video_id"]][key].append((int(d["frame_id"]), float(d["x"]), float(d["y"])))
    for v in out.values():
        v["valid"].sort()
        v["invalid"].sort()
    return out


def load_videos(input_dir: Path) -> tuple[dict, dict]:
    from core.dataset import load_manifest, resolve_manifest
    man = resolve_manifest(str(input_dir))
    videos = load_manifest(man)
    return {v.video_id: v for v in videos}, {v.video_id: v.angle for v in videos}


def load_obs(traj_dir: Path, vid: str) -> list:
    """读 dump 的观测文件 → 按帧号升序的 [(f, x, y, conf)]（已按帧融合）。"""
    per_frame: dict[int, list[tuple[float, float, float]]] = {}
    for line in (traj_dir / f"traj_{vid}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        per_frame.setdefault(int(d["f"]), []).append((float(d["x"]), float(d["y"]), float(d["c"])))
    fused = fuse_obs(per_frame)
    return sorted((f, x, y, c) for f, (x, y, c) in fused.items())


# ── 判定重放（唯一实现 = landing.analyze_landings）────────────────────────
def replay(points: list, meta, cfg: dict) -> list:
    """按 solution._analyze 的完整管线重放 → [(frame_id, x, y)] 预测。

    直接调用与提交侧相同的 ``landing.analyze_landings``，不存在镜像拷贝，
    离线评分与线上行为逐位一致。
    """
    det_cfg = cfg["detector"]
    conf_gate = float(det_cfg.get("conf_gate", 0.6))
    land_cfg = cfg["landing"]
    merged = _solution._deep_merge(land_cfg.get("default", {}),
                                   land_cfg.get(meta.angle, {}))
    return analyze_landings(points, meta.angle, meta.width, meta.height,
                            meta.n_frames, conf_gate, merged)


# ── 官方口径评分（Kuhn 最大基数匹配）────────────────────────────────────────
def max_matching(adj: list, n_right: int) -> list:
    """Kuhn 最大基数匹配；返回长度 n_right 的 match（右侧点 -> 左侧点）。"""
    match_r = [-1] * n_right

    def dfs(u: int, seen: list) -> bool:
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


def match_counts(P: list, G: list) -> tuple[int, int, int, int]:
    """单视频匹配，返回 (tp, fp, fn, ignored)。G = [(fid,x,y,valid)]。"""
    adj = [[] for _ in P]
    for i, (pf, px, py) in enumerate(P):
        for j, (gf, gx, gy, _v) in enumerate(G):
            if abs(pf - gf) <= HIT_FRAMES and math.hypot(px - gx, py - gy) <= HIT_DIST:
                adj[i].append(j)
    match_r = max_matching(adj, len(G)) if G else []
    matched = {u for u in match_r if u != -1}
    tp = sum(1 for j, u in enumerate(match_r) if u != -1 and G[j][3])
    ign = sum(1 for j, u in enumerate(match_r) if u != -1 and not G[j][3])
    fn = sum(1 for g in G if g[3]) - tp
    fp = len(P) - len(matched)
    return tp, fp, fn, ign


def prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(points_by_vid: dict, gt: dict, videos: dict, cfg: dict) -> dict:
    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "ign": 0, "n_gt": 0} for v in VIEWS}
    for vid, meta in videos.items():
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + \
            [(f, x, y, False) for f, x, y in g["invalid"]]
        preds = replay(points_by_vid[vid], meta, cfg)
        tp, fp, fn, ign = match_counts(preds, G)
        a = acc[meta.angle]
        a["tp"] += tp
        a["fp"] += fp
        a["fn"] += fn
        a["ign"] += ign
        a["n_gt"] += len(g["valid"])
    score = 0.0
    for v in VIEWS:
        a = acc[v]
        a["p"], a["r"], a["f1"] = prf1(a["tp"], a["fp"], a["fn"])
        score += 25.0 * a["f1"]
    acc["score"] = score
    return acc


def format_report(acc: dict, cfg: dict, title: str = "") -> str:
    out = []
    if title:
        out.append(title)
    out.append(f"参数: {json.dumps(cfg, ensure_ascii=False)}")
    out.append(f"{'视角':<6}{'GT':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'忽略':>6}"
               f"{'P':>8}{'R':>8}{'F1':>8}{'折分/25':>9}")
    for v in VIEWS:
        a = acc[v]
        out.append(f"{v:<6}{a['n_gt']:>5}{a['tp']:>5}{a['fp']:>5}{a['fn']:>5}{a['ign']:>6}"
                   f"{a['p']:>8.3f}{a['r']:>8.3f}{a['f1']:>8.3f}{25 * a['f1']:>9.2f}")
    out.append(f"{'合计':<6}{'':>5}{'':>5}{'':>5}{'':>5}{'':>6}{'':>8}{'':>8}{'':>8}"
               f"{acc['score']:>9.2f}")
    return "\n".join(out)


# ── dump：跑一遍模型，落盘逐帧观测 ──────────────────────────────────────────
def _ort_infer(onnx_path: Path, ort_dir=None, log=print):
    if ort_dir:
        sys.path.insert(0, str(ort_dir))
    import onnxruntime as ort
    avail = ort.get_available_providers()
    # 推理后端自动选：有 CUDA 用 CUDA，否则 CPU（不再与解码后端混淆）
    want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    if not want:
        want = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(onnx_path), so, providers=want)
    in_name = sess.get_inputs()[0].name
    log(f"[dump] ort providers={sess.get_providers()} in={in_name}")

    def infer(x):
        return sess.run(None, {in_name: x})[0]
    return infer


def _trt_infer(onnx_path: Path, cache: Path, log=print):
    from core.trt import load_or_build
    model, how = load_or_build(onnx_path, cache, input_name="input",
                               shape=(1, 9, IN_H, IN_W), log=log)
    log(f"[dump] tensorrt {how}")
    return model.infer


def iter_frames_cv2(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 打不开 {path}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def cmd_dump(a) -> int:
    videos, _ = load_videos(Path(a.input))
    input_dir = Path(a.input)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = ROOT / "participant" / "weights" / "ball.onnx"

    # --decoder 是 --decode 的向后兼容别名
    decode = a.decoder if getattr(a, "decoder", None) else a.decode

    # 推理后端（ort/trt）与解码后端（--decode）彻底分离
    if a.backend == "ort":
        infer = _ort_infer(onnx, a.ort_dir, log=print)
    else:
        cache = Path(a.cache) if a.cache else Path(a.out) / "trt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        infer = _trt_infer(onnx, cache, log=print)

    # 复用提交的 BallDetector 配置（与 solution.prepare 的构造逻辑一致）
    det_cfg = {k: v for k, v in _solution.load_config()["detector"].items()
               if k in ("score_th", "channels", "peak_radius")}
    chs = tuple(c for c in det_cfg.get("channels", [1]) if c in (0, 1, 2)) or (1,)
    det_cfg["channels"] = chs
    det = BallDetector(infer, onnx, **det_cfg)

    ids = list(videos)
    if a.limit:
        ids = ids[:a.limit]
    for vid in ids:
        meta = videos[vid]
        det.reset()
        n = 0
        with (out_dir / f"traj_{vid}.jsonl").open("w", encoding="utf-8") as fh:
            for fid, img in _iter_frames(meta, input_dir, decode):
                for off, x, y, c in det.detect(img, meta.width, meta.height):
                    fh.write(json.dumps({"f": fid + off, "x": x, "y": y, "c": c,
                                         "offset": off}) + "\n")
                n += 1
        print(f"[dump] {vid} [{meta.angle}] {n} 帧 -> traj_{vid}.jsonl", flush=True)
    print(f"[dump] 完成。评分：python tune_eval.py score --traj {out_dir} --input {a.input}")
    return 0


def _iter_frames(meta, input_dir: Path, mode: str):
    """解码帧迭代器，``mode`` 为视频解码后端（cv2/cpu/gpu），yield ``(frame_id, rgb)``。

    - ``gpu``：ffmpeg NVDEC（与线上评测一致）
    - ``cpu``：PyAV 直读
    - ``cv2``：opencv（本地无 ffmpeg/av 时的兜底，BGR→RGB）
    """
    if mode == "cv2":
        yield from enumerate(iter_frames_cv2(input_dir / meta.path))
    else:
        from core.decoder import decode_video
        yield from decode_video(meta, input_dir, mode)


# ── score ───────────────────────────────────────────────────────────────────
def cmd_score(a) -> int:
    gt_path = Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl"
    gt = load_gt(gt_path)
    videos, _ = load_videos(Path(a.input))
    traj = Path(a.traj)
    missing = [v for v in videos if not (traj / f"traj_{v}.jsonl").exists()]
    if missing:
        print(f"缺少轨迹文件（先跑 dump）：{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        return 1
    points = {vid: load_obs(traj, vid) for vid in videos}

    # 与提交侧完全相同的配置（读 configs/landing.json + 内置默认合并）
    cfg = _solution.load_config()

    # 按视频分视角评估（提交也是按视角取配置）
    acc = {v: {"tp": 0, "fp": 0, "fn": 0, "ign": 0, "n_gt": 0} for v in VIEWS}
    for vid, meta in videos.items():
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + \
            [(f, x, y, False) for f, x, y in g["invalid"]]
        preds = replay(points[vid], meta, cfg)
        tp, fp, fn, ign = match_counts(preds, G)
        st = acc[meta.angle]
        st["tp"] += tp
        st["fp"] += fp
        st["fn"] += fn
        st["ign"] += ign
        st["n_gt"] += len(g["valid"])
        if a.per_video:
            p, r, f1 = prf1(tp, fp, fn)
            print(f"  {vid:<10}{meta.angle:<6}{len(g['valid']):>4}{tp:>4}{fp:>4}{fn:>4}"
                  f"{ign:>6}{p:>8.3f}{r:>8.3f}{f1:>8.3f}")
    score = 0.0
    for v in VIEWS:
        st = acc[v]
        st["p"], st["r"], st["f1"] = prf1(st["tp"], st["fp"], st["fn"])
        score += 25.0 * st["f1"]
    acc["score"] = score
    print(format_report(acc, {}, "\n官方口径（Kuhn 匹配）"))
    return 0


# ── diag：FN/FP 逐例局部轨迹诊断 ──────────────────────────────────────────
def _kink_diag(points, meta, land_cfg, conf_gate):
    """逐步重放判定管线，返回各阶段中间结果，用于 FN/FP 归因。"""
    from participant.landing import (clean_track, find_landings, keep_in_roi,
                                     rally_filter, table_roi, kink_candidates)
    gated = [p for p in points if p[3] >= conf_gate]
    pts = gated if len(gated) >= 3 else points

    cleaned = clean_track(pts,
                          max_resid=float(land_cfg.get("max_resid", 30.0)),
                          max_gap=int(land_cfg.get("max_gap", 6)),
                          mode=str(land_cfg.get("clean_mode", "chord")))
    cands = kink_candidates(cleaned, int(land_cfg.get("k", 2)),
                            float(land_cfg.get("span", 6)),
                            float(land_cfg.get("max_step", 2.0)))
    raw = find_landings(cleaned, land_cfg)
    rois = table_roi(land_cfg.get("roi"), meta.width, meta.height)
    kept = keep_in_roi(raw, rois, min_keep=float(land_cfg.get("roi_min_keep", 0.5)))
    final = rally_filter(cleaned, kept, rois, angle=meta.angle)
    return {"gated": pts, "cleaned": cleaned, "cands": cands,
            "raw": raw, "roi": rois, "final": final}


def _reasons(c, land_cfg):
    """返回候选 c 被各门控拒绝的原因列表（空列表 = 通过全部门控）。"""
    r = []
    dv_y = float(land_cfg.get("dv_y", -10.0))
    max_dvx_ratio = float(land_cfg.get("max_dvx_ratio", 0.5))
    dv_strong = float(land_cfg.get("dv_strong", 0.0))
    ratio_strong = float(land_cfg.get("ratio_strong", 0.0))
    if c[3] > dv_y:
        r.append(f"Δv_y={c[3]:.1f}>{dv_y}")
    dir_ok = abs(c[4]) <= max_dvx_ratio * abs(c[3])
    strong_ok = (dv_strong > 0 and abs(c[3]) >= dv_strong
                 and abs(c[4]) <= ratio_strong * abs(c[3]))
    if not (dir_ok or strong_ok):
        r.append(f"|Δv_x|={abs(c[4]):.1f}超标")
    conf_min = land_cfg.get("conf_min")
    if conf_min is not None and c[5] < float(conf_min):
        r.append(f"conf={c[5]:.2f}<{conf_min}")
    vb_min = land_cfg.get("vb_min")
    if vb_min is not None and c[6] < float(vb_min):
        r.append(f"v_y前={c[6]:.1f}<{vb_min}")
    va_max = land_cfg.get("va_max")
    if va_max is not None and c[7] > float(va_max):
        r.append(f"v_y后={c[7]:.1f}>{va_max}")
    dt_max = land_cfg.get("dt_max")
    if dt_max is not None and (c[8] > float(dt_max) or c[9] > float(dt_max)):
        r.append(f"dt={c[8]:.0f}/{c[9]:.0f}>{dt_max}")
    return r


def _print_window(points, cands, diag, gf, win, land_cfg):
    print(f"  原始轨迹点（f±{win}）：")
    n = 0
    for p in points:
        if abs(p[0] - gf) <= win:
            n += 1
            print(f"    f={p[0]:.0f} x={p[1]:.1f} y={p[2]:.1f} conf={p[3]:.2f}")
    if n == 0:
        print("    （无）")
    print(f"  折角候选（f±{win}，Δv_y 单位 px/帧）：")
    found = False
    for c in cands:
        if abs(c[0] - gf) <= win:
            found = True
            rs = _reasons(c, land_cfg)
            mark = "KEEP" if not rs else "REJ:" + ";".join(rs)
            print(f"    f={c[0]:.0f} x={c[1]:.1f} y={c[2]:.1f} Δv_y={c[3]:.1f} "
                  f"Δv_x={c[4]:.1f} conf={c[5]:.2f} v前={c[6]:.1f} v后={c[7]:.1f} "
                  f"dt={c[8]:.0f}/{c[9]:.0f} -> {mark}")
    if not found:
        print("    （无折角候选）")
    # 最终落点（经 ROI/rally 后）
    fl = [l for l in diag["final"] if abs(l[0] - gf) <= win]
    print(f"  最终落点（f±{win}）：{[(l[0], round(l[1]), round(l[2])) for l in fl] if fl else '（无）'}")


def cmd_diag(a) -> int:
    gt_path = Path(a.gt) if a.gt else Path(a.input) / "gt_public.jsonl"
    gt = load_gt(gt_path)
    videos, _ = load_videos(Path(a.input))
    traj = Path(a.traj)
    cfg = _solution.load_config()
    win = a.window
    only = set(a.only.split(",")) if a.only else None

    for vid, meta in videos.items():
        if only and vid not in only:
            continue
        g = gt.get(vid, {"valid": [], "invalid": []})
        G = [(f, x, y, True) for f, x, y in g["valid"]] + \
            [(f, x, y, False) for f, x, y in g["invalid"]]
        points = load_obs(traj, vid)
        det_cfg = cfg["detector"]
        conf_gate = float(det_cfg.get("conf_gate", 0.6))
        land_cfg = _solution._deep_merge(cfg["landing"].get("default", {}),
                                         cfg["landing"].get(meta.angle, {}))
        preds = replay(points, meta, cfg)
        tp, fp, fn, ign = match_counts(preds, G)

        # 重新算 Kuhn 匹配，拿到「哪些 GT 被命中 / 哪些预测是孤立 FP」
        adj = [[] for _ in preds]
        for i, (pf, px, py) in enumerate(preds):
            for j, (gf, gx, gy, _v) in enumerate(G):
                if abs(pf - gf) <= HIT_FRAMES and math.hypot(px - gx, py - gy) <= HIT_DIST:
                    adj[i].append(j)
        match_r = max_matching(adj, len(G)) if G else []
        matched_gt = {j for j, u in enumerate(match_r) if u != -1}
        matched_pred = {u for u in match_r if u != -1}

        if fn == 0 and fp == 0:
            continue
        print(f"\n===== {vid} [{meta.angle}] TP={tp} FP={fp} FN={fn} "
              f"n_frames={meta.n_frames} =====")
        print(f"  全部预测：{[(p[0], round(p[1]), round(p[2])) for p in preds]}")
        diag = _kink_diag(points, meta, land_cfg, conf_gate)
        cands = diag["cands"]

        for j, (gf, gx, gy, gv) in enumerate(G):
            if not gv or j in matched_gt:
                continue
            print(f"\n[FN] GT frame={gf} x={gx:.1f} y={gy:.1f}")
            _print_window(points, cands, diag, gf, win, land_cfg)
        for i in range(len(preds)):
            if i not in matched_pred:
                pf, px, py = preds[i]
                print(f"\n[FP] pred frame={pf} x={px:.1f} y={py:.1f}")
                _print_window(points, cands, diag, pf, win, land_cfg)
    return 0


# ── selftest：不依赖模型/视频的落点逻辑冒烟测试 ──────────────────────────
def cmd_selftest(a) -> int:
    """验证边缘候选与判定管线的关键行为（纯 numpy，秒级）。"""
    from participant.landing import analyze_landings, _edge_candidates

    fails = 0
    cfg = {"landing": {"default": {"edge": {"start_min_rise": 12.0,
                                           "start_vy_max": -3.0,
                                           "start_conf": 0.7,
                                           "start_max_frame": 4},
                                    "roi": None, "clean_mode": "chord",
                                    "max_resid": 30.0, "max_gap": 6,
                                    "k": 2, "span": 4.0, "max_step": 1.0,
                                    "dv_y": -12.0, "max_dvx_ratio": 0.5,
                                    "min_gap": 6.0, "nms": "maxy",
                                    "roi_min_keep": 0.5}}}
    lc = cfg["landing"]["default"]

    # 1) 起始边缘：开头连续抬升 → 应产生候选
    rising = [(2, 1032.0, 568.6, 0.87), (3, 1042.0, 567.0, 0.90),
              (4, 1050.0, 541.7, 0.89), (5, 1057.0, 524.6, 0.77)]
    ec = _edge_candidates(rising, lc)
    assert len(ec) == 1 and ec[0][0] == 2, f"起始边缘应命中 f2，实际 {ec}"
    print(f"[selftest] 起始边缘 OK: {ec}")

    # 2) 起始边缘：开头连续下降 → 不应产生候选
    falling = [(0, 900.0, 400.0, 0.9), (1, 900.0, 430.0, 0.9),
               (2, 900.0, 460.0, 0.9), (3, 900.0, 490.0, 0.9)]
    assert _edge_candidates(falling, lc) == [], "下降开头不应产生候选"
    print("[selftest] 下降开头不误报 OK")

    # 3) 首帧帧号超过 start_max_frame → 不产生候选
    late = [(10, 1032.0, 568.6, 0.87), (11, 1042.0, 567.0, 0.90),
            (12, 1050.0, 541.7, 0.89), (13, 1057.0, 524.6, 0.77)]
    assert _edge_candidates(late, lc) == [], "非开头帧不应产生边缘候选"
    print("[selftest] 非开头不误报 OK")

    # 4) 完整管线对空轨迹/单点轨迹稳健
    assert analyze_landings([], "0度", 1920, 1080, 100, 0.5, lc) == []
    assert analyze_landings([(0, 1.0, 1.0, 0.9)], "0度", 1920, 1080, 100, 0.5, lc) == []
    print("[selftest] 空/单点轨迹稳健 OK")

    print("[selftest] 全部通过" if not fails else f"[selftest] {fails} 处失败")
    return 0 if not fails else 1


# ── CLI ─────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="落点检测离线精度评测（开发工具）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="跑一遍模型，落盘逐帧观测")
    d.add_argument("--input", required=True)
    d.add_argument("--out", default="/tmp/pp_traj")
    # 视频解码后端与推理后端分离：--decode 管解码（gpu=NVDEC 与线上一致），
    # --backend 管推理（trt/ort）。
    d.add_argument("--decode", choices=["gpu", "cpu", "cv2"], default="gpu",
                   help="视频解码后端（默认 gpu，与线上评测一致）")
    d.add_argument("--decoder", choices=["gpu", "cpu", "cv2"], default=None,
                   help="已废弃：--decode 的别名（向后兼容）")
    d.add_argument("--backend", choices=["trt", "ort"], default="ort",
                   help="推理后端")
    d.add_argument("--cache", default=None)
    d.add_argument("--ort-dir", default=None)
    d.add_argument("--limit", type=int, default=0)
    d.set_defaults(func=cmd_dump)

    s = sub.add_parser("score", help="离线重放 + 官方口径评分")
    s.add_argument("--traj", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--gt", default=None)
    s.add_argument("--per-video", action="store_true")
    s.set_defaults(func=cmd_score)

    g = sub.add_parser("diag", help="对 FN/FP 逐例输出局部轨迹诊断")
    g.add_argument("--traj", required=True)
    g.add_argument("--input", required=True)
    g.add_argument("--gt", default=None)
    g.add_argument("--window", type=int, default=8)
    g.add_argument("--only", default=None, help="只诊断指定 video_id（逗号分隔）")
    g.set_defaults(func=cmd_diag)

    st = sub.add_parser("selftest", help="落点逻辑冒烟测试（不需要模型/视频）")
    st.set_defaults(func=cmd_selftest)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
