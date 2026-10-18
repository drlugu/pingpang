"""landing.py 核心函数单元测试（纯 numpy，无模型/视频/GPU）。

运行:  python test_landing.py
"""
import sys
from pathlib import Path

_TASK = Path(__file__).resolve().parent / "participant" / "pingpang"
sys.path.insert(0, str(_TASK))  # `participant` 包在此目录下
sys.path.insert(0, str(_TASK / "participant"))

from participant.landing import (fuse_obs, clean_track, kink_candidates,
                                 pick_landings, rally_filter)

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        raise SystemExit(1)


def test_fuse_obs():
    print("== fuse_obs ==")
    out = fuse_obs({0: [(10.0, 20.0, 0.9)]})
    check("单观测透传", out[0] == (10.0, 20.0, 0.9), str(out))
    out = fuse_obs({0: [(10.0, 20.0, 0.9), (12.0, 20.0, 0.7), (11.0, 20.0, 0.8)]})
    check("多通道均值+最大置信", abs(out[0][0] - 11.0) < 0.01 and abs(out[0][2] - 0.9) < 0.01, str(out))
    out = fuse_obs({0: [(10.0, 20.0, 0.95), (100.0, 200.0, 0.6)]})
    check("离群通道过滤", abs(out[0][0] - 10.0) < 0.01, str(out))
    out = fuse_obs({0: [], 1: [(5.0, 5.0, 0.5)]})
    check("空观测跳过", 0 not in out and 1 in out, str(out))


def test_clean_track():
    print("== clean_track ==")
    pts = [(0, 0.0, 0.0, 0.95), (1, 10.0, 0.0, 0.8), (2, 20.0, 0.0, 0.95),
           (3, 30.0, 50.0, 0.5), (4, 40.0, 0.0, 0.95), (5, 50.0, 0.0, 0.95)]
    out = clean_track(pts, max_resid=10.0, max_gap=6, rounds=2, mode="chord")
    check("chord 删离群点", 3 not in [p[0] for p in out], str([p[0] for p in out]))
    pts2 = [(0, 0.0, 0.0, 0.95), (1, 10.0, 0.0, 0.95), (2, 20.0, 50.0, 0.95), (3, 30.0, 0.0, 0.95)]
    out2 = clean_track(pts2, max_resid=10.0, max_gap=6, rounds=2, mode="chord")
    check("高置信度离群点保留", 2 in [p[0] for p in out2], str([p[0] for p in out2]))
    check("空轨迹", clean_track([], 10, 6, mode="chord") == [])
    check("单点轨迹", clean_track([(0, 1.0, 1.0, 0.9)], 10, 6, mode="chord") == [(0, 1.0, 1.0, 0.9)])
    # side 模式保留折角顶点（触台帧偏离插值弦）
    pts3 = [(0, 0.0, 100.0, 0.9), (1, 0.0, 110.0, 0.9), (2, 0.0, 130.0, 0.85),
            (3, 0.0, 120.0, 0.9), (4, 0.0, 100.0, 0.9)]
    out3 = clean_track(pts3, max_resid=20.0, max_gap=6, rounds=2, mode="side")
    check("side 保留折角顶点", 2 in [p[0] for p in out3], str([p[0] for p in out3]))


def test_kink_candidates():
    print("== kink_candidates ==")
    pts = [(0, 0.0, 100.0, 0.9), (1, 0.0, 110.0, 0.9), (2, 0.0, 120.0, 0.9),
           (3, 0.0, 115.0, 0.9), (4, 0.0, 100.0, 0.9)]
    c = kink_candidates(pts, k=2, span=4.0, max_step=2.0)
    check("产生折角候选", len(c) > 0, str(c))
    f2 = [x for x in c if x[0] == 2.0]
    check("折角在 y 峰值且 Δv_y<0", len(f2) == 1 and f2[0][3] < 0, str(f2))
    pts_gap = [(0, 0.0, 100.0, 0.9), (1, 0.0, 110.0, 0.9), (5, 0.0, 120.0, 0.9),
               (6, 0.0, 110.0, 0.9), (7, 0.0, 100.0, 0.9)]
    check("断口不产生候选", len(kink_candidates(pts_gap, k=2, span=6.0, max_step=1.0)) == 0)
    pts_short = [(0, 0.0, 100.0, 0.9), (1, 0.0, 110.0, 0.9)]
    check("短轨迹无候选", len(kink_candidates(pts_short, k=2, span=4.0, max_step=2.0)) == 0)


def test_pick_landings():
    print("== pick_landings ==")
    args = dict(dv_y=-12.0, max_dvx_ratio=0.5, min_gap=6.0, nms="maxy",
                dv_strong=0.0, ratio_strong=0.0, vb_min=None, va_max=None,
                dt_max=None, conf_min=None)
    pk = pick_landings([(5.0, 100.0, 200.0, -20.0, 2.0, 0.9, 10.0, -10.0, 1.0, 1.0)], **args)
    check("通过全部门控", len(pk) == 1, str(pk))
    pk2 = pick_landings([(5.0, 100.0, 200.0, -8.0, 2.0, 0.9, 10.0, 2.0, 1.0, 1.0)], **args)
    check("Δv_y 不足拒绝", len(pk2) == 0, str(pk2))
    pk3 = pick_landings([(5.0, 100.0, 200.0, -20.0, 30.0, 0.9, 10.0, -10.0, 1.0, 1.0)], **args)
    check("方向不一致拒绝", len(pk3) == 0, str(pk3))
    args_vb = dict(args, vb_min=0.0)
    pk4 = pick_landings([(5.0, 100.0, 200.0, -20.0, 2.0, 0.9, -5.0, -25.0, 1.0, 1.0)], **args_vb)
    check("vb_min 拒绝", len(pk4) == 0, str(pk4))
    cands2 = [(5.0, 100.0, 200.0, -20.0, 2.0, 0.9, 10.0, -10.0, 1.0, 1.0),
              (6.0, 105.0, 200.0, -22.0, 2.0, 0.8, 10.0, -12.0, 1.0, 1.0)]
    check("NMS 去重", len(pick_landings(cands2, **args)) == 1)


def test_rally_filter():
    print("== rally_filter ==")
    pts = [(i, 100.0, 100.0, 0.9) for i in range(12)] + [(12, 200.0, 300.0, 0.9)]
    out = rally_filter(pts, [(12, 200.0, 300.0, 1.0)], angle="45度")
    check("静态阶段后抑制", len(out) == 0, str(out))
    check("空落点", rally_filter(pts, [], angle="45度") == [])
    # 二次弹跳：相邻落点近 + 后者弱 → 抑制后者
    pts3 = [(i, 0.0, 120 - 5 * i, 0.9) for i in range(8)]
    out3 = rally_filter(pts3, [(2, 0.0, 110.0, 1.0), (5, 0.0, 95.0, 1.0)], angle="45度")
    check("二次弹跳抑制后者", len(out3) == 1 and out3[0][0] == 2, str(out3))


def main():
    test_fuse_obs()
    test_clean_track()
    test_kink_candidates()
    test_pick_landings()
    test_rally_filter()
    print(f"\n全部通过（{PASS} 个断言）")


if __name__ == "__main__":
    main()
