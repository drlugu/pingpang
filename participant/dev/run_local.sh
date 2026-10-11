#!/usr/bin/env bash
# 本地复跑入口：绕开 run.sh 里写死的评测平台路径。
#
# 为什么不能直接 ./run.sh ？官方入口是给平台镜像写的，里面写死了：
#   PY=/opt/conda/envs/conda_env/bin/python3      （本地不存在，有 fallback 不致命）
#   export PYTHONPATH=/participant/pingpang       （本地错路径）
#   "$PY" /participant/pingpang/integrity.py -v   （本地不存在 → set -e 直接退出，码 2）
#   exec   /participant/pingpang/run.py ...       （同上）
#   INPUT=${1:-/participant/input} OUTPUT=${1:-/participant/output}   （本地无此目录）
#   DECODER=${DECODER:-gpu}                       （本机无 GPU / 无 torch / 无 tensorrt）
# 本机实测：/usr/bin/python3 3.10.12 + onnxruntime 1.23.2 + cv2 4.10.0 + numpy 1.26.4；
#           无 torch、无 tensorrt、无 nvidia-smi。故本地一律 --decoder cpu（走 weights/ball.onnx）。
#
# 用法（任意当前目录）：
#   bash participant/dev/run_local.sh            # 全量 15 条 + 评分（CPU ≈ 13 分钟）
#   bash participant/dev/run_local.sh --check    # 只跑 1 条小视频自检（≈ 40 秒）
#   OUT=/tmp/out_mine DECODER=cpu bash participant/dev/run_local.sh
#
# 正式评测（官方镜像内有 conda_env + TRT + GPU）仍用 --decoder gpu。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 仓库根（scripts/core/participant 所在处）
cd "$ROOT"

DECODER="${DECODER:-cpu}"
INPUT="${INPUT:-public_data}"        # 数据集根：manifest.json + videos/
OUT="${OUT:-/tmp/out_pingpang}"
SOLUTION="participant.solution:Solution"

if [[ "${1:-}" == "--check" ]]; then
    # 1 条视频端到端：解码→算法→写出→schema 校验，暴露接入类错误
    exec python3 scripts/check_solution.py \
        --input "$INPUT" --solution "$SOLUTION" --decoder "$DECODER"
fi

echo "[run_local] decoder=$DECODER input=$INPUT output=$OUT"
python3 run.py --input "$INPUT" --output "$OUT" \
    --solution "$SOLUTION" --decoder "$DECODER"

echo "[run_local] 评分（官方口径复刻：帧容差±1 / 位置容差 50px / 忽略 valid=false）"
exec python3 participant/dev/eval_f1.py --predictions "$OUT/predictions.jsonl"
