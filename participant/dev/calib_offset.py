#!/usr/bin/env python3
"""标定：检测模型输出通道 ↔ 时间帧偏移（用真值落点做锚点）。

    python3 participant/dev/calib_offset.py --input public_data --limit 6

对每个真值落点 (video, frame g, x, y)：把"以帧 f 结尾的三帧栈"喂给模型，得到 3 个
通道各自的峰值位置（反算回原分辨率）。若通道 c 编码的是物理帧 f+delta_c，那么
用 f = g - delta_c 的那次推理，峰值应当落在真值 (x, y) 附近。对 delta 网格
-2..+2 统计峰值到真值的**中位距离**，最小值对应的 delta 就是该通道的真实偏移。

输出即 configs/landing.json 里 detector.channels 的含义：通道 c ↔ 相对当前帧
delta_c。这一步决定观测被归到哪一帧，错 1 帧就正好吃掉 ±1 帧的匹配容差。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dataset import load_manifest, resolve_manifest  # noqa: E402
from core.decoder import decode_video  # noqa: E402
from participant.ball_detector import IN_H, IN_W, BallDetector  # noqa: E402
from participant.solution import _HERE, _onnx_session  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检测模型通道-帧偏移标定")
    ap.add_argument("--input", default="public_data")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--gt", default="public_data/gt_public.jsonl")
    ap.add_argument("--limit", type=int, default=6, help="只用前 N 条视频（省时间）")
    ap.add_argument("--decoder", default="cpu", choices=["cpu", "gpu"])
    a = ap.parse_args(argv)

    gt = defaultdict(list)
    for line in Path(a.gt).open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("valid", True):
                gt[r["video_id"]].append((int(r["frame_id"]), float(r["x"]), float(r["y"])))

    onnx_path = _HERE / "weights" / "ball.onnx"
    sess = _onnx_session(onnx_path, a.decoder)
    in_name = sess.get_inputs()[0].name

    peaks: dict[tuple[str, int, int], tuple[float, float]] = {}
    peak_val: dict[tuple[str, int, int], float] = {}
    for meta in load_manifest(resolve_manifest(a.input, a.manifest))[:a.limit]:
        marks = gt.get(meta.video_id, [])
        if not marks:
            continue
        want = set()
        for g, _x, _y in marks:
            want.update(range(g - 3, g + 4))
        buf: list[np.ndarray] = []
        n_inf = 0
        for fid, img in decode_video(meta, a.input, a.decoder):
            buf.append(BallDetector._normalize(img))
            buf = buf[-3:]
            if len(buf) < 3 or fid not in want:
                continue
            inp = np.concatenate(buf, axis=0)[None]
            heat = sess.run(None, {in_name: np.ascontiguousarray(inp)})[0]
            heat = heat[0] if heat.ndim == 4 else heat     # 去掉 batch 维 → (3,H,W)
            n_inf += 1
            for c in range(heat.shape[0]):
                h = heat[c]
                py, px = divmod(int(np.argmax(h)), h.shape[1])
                peaks[(meta.video_id, fid, c)] = (
                    (px + 0.5) * meta.width / IN_W, (py + 0.5) * meta.height / IN_H)
                peak_val[(meta.video_id, fid, c)] = float(h.max())
        print(f"{meta.video_id} [{meta.angle}] {meta.n_frames} 帧，推理 {n_inf} 次，"
              f"真值 {len(marks)} 个，峰值分位 "
              f"{np.percentile(list(peak_val.values()), [5, 50, 95]).round(3)}")

    if not peaks:
        print("没有任何标定数据（--limit 太小？）")
        return 1

    print(f"\n{'通道':<4}" + "".join(f"delta={d:>3}" for d in range(-3, 4)))
    for c in range(3):
        row = []
        for d in range(-3, 4):
            ds = []
            for vid, marks in gt.items():
                for g, gx, gy in marks:
                    p = peaks.get((vid, g - d, c))     # 以帧 g-d 结尾的栈
                    if p is not None:
                        ds.append(((p[0] - gx) ** 2 + (p[1] - gy) ** 2) ** 0.5)
            row.append(np.median(ds) if ds else float("nan"))
        finite = [v for v in row if not np.isnan(v)]
        best = (int(np.nanargmin(row)) - 3) if finite else None
        print(f"ch{c:<2}" + "".join(f"{v:9.1f}" for v in row)
              + (f"   → 最佳 delta={best:+d}" if best is not None else "   → 无数据"))
    print("\n解读：某通道的最优 delta 就是它编码的物理帧相对「栈尾帧 f」的偏移；"
          "中位距离越小越可信（>50px 基本等于没对准）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
