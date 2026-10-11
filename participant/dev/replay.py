#!/usr/bin/env python3
"""从落盘的逐帧观测**重放生产代码的落点分析链路**，秒级出 predictions.jsonl + 分数。

为什么可信：落点分析是纯后处理（``Solution._analyze`` 只吃逐帧观测），
推理/解码的结果已经被 PP_DUMP_DET 的 dump 完整捕获，因此
「重放结果」与「真跑一遍推理的结果」逐字节一致。改配置后不必重跑推断。

    python3 participant/dev/replay.py --detections <out>/detections \\
        --out /tmp/predictions.jsonl --config participant/configs/landing.json

脚本直接实例化 ``participant.solution.Solution`` 并调用其 ``prepare``（加载真实配置）
与 ``_analyze``（真实算法），因此**测的是生产代码本身**，不是算法的副本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from core.types import RunContext, VideoMeta  # noqa: E402
from participant.dev import eval_f1  # noqa: E402
from participant.solution import Solution  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="落点分析链路重放")
    ap.add_argument("--detections", required=True, help="PP_DUMP_DET=1 产出的 dump 目录")
    ap.add_argument("--config", default=str(_ROOT / "participant" / "configs"
                                           / "landing.json"))
    ap.add_argument("--out", default="/tmp/replay/predictions.jsonl")
    ap.add_argument("--gt", default=str(_ROOT / "public_data" / "gt_public.jsonl"))
    ap.add_argument("--manifest", default=str(_ROOT / "public_data" / "manifest.json"))
    ap.add_argument("--engine-dir", default="/tmp/replay/engine",
                    help="prepare 的 output_dir（TRT 缓存落这里，别污染正式输出）")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)

    sol = Solution()
    sol.prepare(RunContext(input_dir=str(_ROOT), output_dir=a.engine_dir,
                           decoder="cpu", info={}))

    docs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(a.detections).glob("*.json"))]
    if not docs:
        raise SystemExit(f"{a.detections} 下没有 *.json —— 先 PP_DUMP_DET=1 跑一次推理")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for doc in docs:
            meta = VideoMeta(video_id=doc["video_id"], path="", angle=doc["angle"], fps=0,
                             width=int(doc["width"]), height=int(doc["height"]),
                             n_frames=int(doc["n_frames"]))
            sol.reset(meta)
            for p in doc["points"]:      # dump 已是融合后观测，回填即成 idempotent
                sol.obs.setdefault(int(p[0]), []).append((p[1], p[2], p[3]))
            for l in sol.finish_video(meta):
                f.write(json.dumps({"video_id": meta.video_id, **l.to_dict()},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"重放完成：{len(docs)} 条视频 / {n} 个落点 -> {out}")
    if a.no_score:
        return 0
    return eval_f1.main(["--predictions", str(out), "--gt", a.gt,
                         "--manifest", a.manifest])


if __name__ == "__main__":
    sys.exit(main())
