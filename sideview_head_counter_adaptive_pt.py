# -*- coding: utf-8 -*-
"""通用 PC/PyTorch 侧视车窗乘员计数。

本文件由两个既有实现整理而来。检测和窗内跟踪沿用 PyTorch/Ultralytics
流程；最终人数采用无车型先验的保守融合：单帧峰值下限、稳定轨迹、
同帧共存硬约束、单帧独立身份晋升，以及带位置/外观证据的短碎轨拼接。
不会根据前后排、固定座位数或某个特定人数排列直接改写结果。

思路
====
在 sideview_headshoulder_counter.py 的 MotionCoherentTracker 内核外面，加一层"窗管理"：

    帧 → 检测 car / window / head
     ①  主车绑定（流式简化版：与上一帧主车中心最近的 car 框）
     ②  全局 shift 每帧只算一次（车框位移，含左右裁切处理）→ 分发给所有窗
     ③  WindowIdTracker 分配跨帧稳定的窗号
     ④  窗过滤（须 ≥80% 被主车覆盖）+ head 分窗（落进哪个窗就归哪个）
     ⑤  每窗一个 MotionCoherentTracker 实例（共享全局 shift）
     ⑥  人数 = Σ 每窗通用融合身份数     ← 正式口径

同时保留原始 distinct 和整车 tracker 作为诊断对照，便于区分检测漏检、
轨迹碎裂与跨窗重复。

为什么运动补偿要"全局算一次"
--------------------------
原 counter 的 RANSAC 需要 >=2 个 (轨迹, 检测) 点对才启用（`shift_inliers >= 2`）。
若照字面"每窗各估一次"，多数窗里只有 1~3 个人 → RANSAC 频繁失效、退化成速度外推。
整车是一起动的，位移是同一个量，所以在整车级估一次再分发；RANSAC 保留为兜底
（车框丢失时自动回退）。

与 nvidia_sideview_dualgate.py 的关系
-----------------------------------
- `WindowIdTracker` 照搬自 514–757 行，但**已有一处刻意偏离**（见下节第 2 条）。
- 逐窗 tracker 用**移植版 WindowHeadTracker**（源 972–1917 行的跟踪核心），
  已并入本文件的 `★ 移植块`（块内保持逐行一致，所有改动一律放在块外）。
  实测这是 59 集 59.3% → 69.5% 的**全部**来源：原以为差距在"漏搬窗口压缩/重排序"，
  实测否定 —— 不做压缩、只换 tracker，聚合指标与 noguard 基线逐项一致（41/59、
  MAE 0.356、9/9），逐序列仅 4/57 条差 ±1 且互相抵消。根因是坐标系：
  MotionCoherentTracker 用绝对像素 + 单个全局平移，而 WindowHeadTracker 内部做
  窗内归一化 (u,v) + 速度 EMA + 外观档案 + 窗框不稳/贴边标记。
  `WINHEAD_TRACKER=motion` 可退回旧实现（59 集 35/59 = 59.3%）作对照。
- **不搬任何护栏**（180 条一条都不要），**也不搬**原 `get_count()` 及其 20 多个
  `get_*_count` 缝合器 —— 那部分夹带座位先验（源码注释有「后排应为3人」），
  虽然 59 集能再涨到 84.7%，但不属于本项目要的无先验口径。
  计数一律用裸 distinct（`count_distinct()`）。

本脚本对 dualgate 的刻意偏离（2026-09-14，均在 `★ 移植块` 之外）
--------------------------------------------------------------
三处都因实测暴露的缺陷而改，**移植块本身保持逐行一致**。

1. **主车绑定改用「预测位置」比较** —— `WindowedHeadCounter._pick_main_car`
   原逻辑拿新检出框与**上一帧主车中心**比距离，超 `CAR_SWITCH_MAX_DIST`(400) 即判为
   背景车、沿用旧框。但车快速驶过时单帧位移可达 200~300px，一帧就超界 → 主车框
   **永久冻结**（差距只增不减，无法恢复）→ 窗口匹配失去运动先验、窗号错乱；且尾帧
   窗框落到冻结车框之外，被 `WINDOW_COVER_MIN` 整体过滤，**该帧人头一颗都不计**。
   改为与「上一帧位移外推出的预测位置」比较，正常行驶残差≈0，只有真跳到别的车才拒绝。
   实测序列 `20260714180408088`：修复前帧 3~5 主车框冻结在 cx=366（真车在 766/940/1108）；
   修复后逐帧跟上，W1 恒为最前窗，尾帧恢复计数。
   **收益：37 集 54.1% → 56.8%，59 集 69.5% → 71.2%。**

2. **窗口编号补上左右顺序约束** —— `WindowIdTracker.update` 末尾
   车头方向固定 ⇒ 各窗沿车轴的左右次序固定 ⇒ 编号次序也必须固定。但代价匹配失败后
   走的"按顺序补配对"是**按编号 zip**、既不查距离也不查左右序，会排出逆序
   （实测 `20260714180408088` 帧 3 的 W1/W2 互换即源于此）。
   现做一次排列修正：**只重排继承自上一帧的编号**（不动本帧新开的编号，免得刚出现的
   新窗抢走 W1），消除逆序而不增删任何编号。

3. **后段恢复：V7 身份断裂改为「不自并」 + 认领门限 0.78 → 0.65**（2026-09-14 合入，
   `WHT_R1_SPLIT_ON_BREAK=0` / `WHT_CLAIM_THR=0.78` 可逐项退回）
   移植块的 V7 历史重关联**已经算出** `strong_history_identity_break`
   （`cost≥0.75 且 reproj≥0.25 且 abs≥0.45`），即它**自己断言"这次重关联过于勉强"**，
   但代码随后**仍把该检测并进旧 tid** → 一个 gid 覆盖两个物理人 → **少算**。
   现改为：被判断裂时放弃该配对（`continue`），让检测走认领 / 新建路径。
   同时把阶段二外观认领门限 `_CLAIM_THR` 从 0.78 降到 0.65
   （0.78 在实测的两套集合上**从未触发**，是死参数；再往下降到 0.55/0.45 结果不再变化，
   故 0.65 是"能拿到全部收益的最高门限"）。
   **两者必须同时开** —— R-1 放开的检测会被认领路径接回**正确的过期 gid**，
   而不是开新 id。单独开 R-1 会误伤（59 集 `..026701` 5→6），
   单独降门限拿不到 `..942884` 那条（该检测已被错误的历史重关联吃掉）。
   实测（Σ每窗 distinct）：
     conf=0.35  36 集 21/36 → 22/36（少算 11→10）｜59 集 42/59 → 44/59（多 8→7 少 9→8）
     conf=0.45  36 集 19/36 → 19/36（不变）      ｜59 集 34/59 → 38/59（多 8→7 少 17→14）
   四个格子准确率全部↑或持平，少算与多算全部↓或持平。
   ⚠ 幅度（±1~2 条）在两套集合的样本量下**未达统计显著**（bootstrap 区间含 0），
   采纳依据是**95 个序列零回归**（从未把一条本来正确的改错）。详见桌面改进报告 §二.3 / §六.3。

   **已否定的后续方向（勿重试）**：曾假设"判据用了绝对量、在不同集合上系统性偏移"
   是 36/59 冲突的成因，试了两种相对化都失败 —— `rank` 模式（候选池内次优解）**恒不触发**；
   `scale` 模式（以头直径为尺）**重现了与绝对判据几乎完全重合的取舍曲线**。
   结论：36/59 冲突不是阈值刻度问题，是 V7 信号本身跨集不可迁移。

已按用户要求移除的开关
--------------------
原脚本的 M-A（长期 EMA 特征）/ M-B（位置先验）/ M-D（低分暂存池）三个改进开关
经实测效果不佳，本脚本不再保留；`ENABLE_APPEARANCE` 开关也去掉 —— 外观始终启用。

用法
====
  python sideview_windowhead_counter.py <输入序列目录> [输出目录]
  python sideview_windowhead_counter.py --seq-dir <目录> --head-conf 0.15

作为模块被 run_selfmade_metrics.py 驱动时，用：
  from sideview_windowhead_counter import WindowedHeadCounter, extract_feat
"""

import os
import sys
import glob
import argparse
import csv
import json
import traceback
import numpy as np
import cv2
from collections import deque, defaultdict
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchreid import models
from torchvision import transforms

# ==========================================================
# 1. 配置 /mnt/disk1/wsk/rk3588/ultralytics_yolov8/ultralytics/260907_rk3588_best_3.pt
# ==========================================================
DEFAULT_SEQ_DIR = r"/mnt/disk1/wsk/rk3588/ultralytics_yolov8/ultralytics/data/test_data/dx_test/1020/images"
DEFAULT_OUTPUT_ROOT = r"/mnt/disk1/wsk/rk3588/ultralytics_yolov8/ultralytics/data/test_data/dx_test/1020/results"

MODEL_PATH = (r"/mnt/disk1/wsk/rk3588/ultralytics_yolov8/runs/260907_best3_fp_fn_640_20261026_r2_hardneg/weights/best.pt")
OSNET_HARD_FALSE_IDS_FILE = os.environ.get("OSNET_HARD_FALSE_IDS_FILE", "")
OSNET_HARD_TRUE_IDS_FILE = os.environ.get("OSNET_HARD_TRUE_IDS_FILE", "")
OSNET_HARD_EXPORT_DIR = os.environ.get("OSNET_HARD_EXPORT_DIR", "")
OSNET_SCORE_TRACE = os.environ.get("OSNET_SCORE_TRACE", "0") == "1"

# ===== 多目录批量运行配置（直接修改这里即可） =====
# 输入根目录：下面每个子目录视为一辆车；如果根目录直接有图片，则视为单车
INPUT_DIR = DEFAULT_SEQ_DIR
# 输出根目录：每辆车独立保存到 OUTPUT_DIR/车辆目录名/
OUTPUT_DIR = DEFAULT_OUTPUT_ROOT
SAVE_ANNOTATED_IMAGES = True
# auto: distinguish the historical 8-frame input from a dense/full sequence.
# short/dense can be used for deterministic comparison or emergency fallback.
SEQUENCE_MODE = os.environ.get("SEQUENCE_MODE", "auto").strip().lower()
# 批量处理时某辆车出错，是否继续处理下一辆车
CONTINUE_ON_ERROR = True
# ==============================================

CLS_CAR, CLS_HEAD, CLS_WINDOW = 0, 1, 2

# 检测阈值：head 默认 0.15（与 run_selfmade_metrics.py 的基线口径一致，保证数字可比）；
# car/window 是大目标，用更高阈值，否则误检车框会把全局 shift 估歪。
HEAD_CONF = 0.15
CAR_CONF = 0.50
WINDOW_CONF = 0.50
CAR_CONTEXT_CONF = 0.20  # 低置信车辆框仅用于判断车窗属于主车还是背景车

def _envf(name, default):
    """环境变量覆盖参数（便于批量调参，不影响默认行为）。"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


IMGSZ = int(_envf("WINHEAD_IMGSZ", 640))

MAX_AGE = _envf("WINHEAD_MAX_AGE", 8)          # 窗内轨迹最大存活帧数
SPATIAL_R = _envf("WINHEAD_SPATIAL_R", 70)     # 空间兜底半径
DIST_GATE = _envf("WINHEAD_DIST_GATE", 80)     # 阶段一位置门限
RANSAC_TOL = 30.0                              # 全局位移投票容差
NMS_IOU = _envf("WINHEAD_NMS_IOU", 0.45)       # 检测 NMS 的 IoU 阈值（调低 = 更强去重）
SIM_GATE = _envf("WINHEAD_SIM_GATE", 0.68)     # 阶段一外观硬门（dualgate 为 0.55）
W_MOT_BASE = _envf("WINHEAD_WM_BASE", 0.2)     # 运动权重下限
W_MOT_RANGE = _envf("WINHEAD_WM_RANGE", 0.3)   # 运动权重随置信度的增程
CLAIM_THR = _envf("WINHEAD_CLAIM_THR", 0.76)   # 阶段二认领门限基准
CLAIM_RANGE = _envf("WINHEAD_CLAIM_RANGE", 0.10)

# 通用计数层：只使用轨迹长度、共存、时序连续性、归一化位置和外观。
# 不使用车型、座位容量、固定窗口人数或特定样本规则。
GENERAL_MIN_HITS = int(_envf("GENERAL_MIN_HITS", 2))
GENERAL_STITCH_GAP = int(_envf("GENERAL_STITCH_GAP", 2))
GENERAL_STITCH_UV = _envf("GENERAL_STITCH_UV", 0.14)
GENERAL_STITCH_APP = _envf("GENERAL_STITCH_APP", 0.78)
GENERAL_FRAGMENT_APP = _envf("GENERAL_FRAGMENT_APP", 0.66)
GENERAL_FRAGMENT_PX = _envf("GENERAL_FRAGMENT_PX", 2.20)
GENERAL_GAP_PX = _envf("GENERAL_GAP_PX", 0.85)
GENERAL_GAP_ANCHOR_U = _envf("GENERAL_GAP_ANCHOR_U", 0.20)
GENERAL_ANCHOR_STITCH_MAX_GAP = int(_envf("GENERAL_ANCHOR_STITCH_MAX_GAP", 12))
GENERAL_SINGLETON_GAP_APP = _envf("GENERAL_SINGLETON_GAP_APP", 0.80)
GENERAL_SINGLETON_GAP_PX = _envf("GENERAL_SINGLETON_GAP_PX", 2.00)
GENERAL_SINGLETON_GAP_ANCHOR_U = _envf("GENERAL_SINGLETON_GAP_ANCHOR_U", 0.45)
GENERAL_BREAK_APP = _envf("GENERAL_BREAK_APP", 0.62)
GENERAL_BREAK_PEER_APP = _envf("GENERAL_BREAK_PEER_APP", 0.70)
GENERAL_BREAK_MOTION_RESIDUAL = _envf("GENERAL_BREAK_MOTION_RESIDUAL", 0.65)
GENERAL_REUSE_GAP_MAX = int(_envf("GENERAL_REUSE_GAP_MAX", 2))
GENERAL_SHORT_SINGLETON_CONF = _envf("GENERAL_SHORT_SINGLETON_CONF", 0.35)
GENERAL_SHORT_SINGLETON_DU = _envf("GENERAL_SHORT_SINGLETON_DU", 0.30)
GENERAL_SHORT_SINGLETON_APP = _envf("GENERAL_SHORT_SINGLETON_APP", 0.60)
GENERAL_SINGLETON_PERSON_FACTOR = _envf("GENERAL_SINGLETON_PERSON_FACTOR", 1.50)


def resolve_sequence_mode(requested, image_count, motion_samples):
    """Resolve short/dense without changing the fixed 2/3-frame hit rules."""
    requested = (requested or SEQUENCE_MODE or "auto").strip().lower()
    if requested not in {"auto", "short", "dense"}:
        raise ValueError(
            f"SEQUENCE_MODE must be auto/short/dense, got: {requested!r}")
    samples = [abs(float(dx)) for dx in motion_samples
               if np.isfinite(dx) and abs(float(dx)) >= 0.5]
    median_abs_dx = float(np.median(samples)) if samples else None
    coverage = len(samples) / max(1, int(image_count) - 1)
    if requested == "auto":
        # Eight-frame inputs always remain on the historical path. A sequence
        # is dense when it has substantially more frames, or when 12+ frames
        # sample the same vehicle motion densely enough.
        is_dense = (
            image_count >= 32 or
            (image_count >= 16 and coverage >= 0.45) or
            (image_count >= 12 and coverage >= 0.55 and
             median_abs_dx is not None and median_abs_dx <= 25.0)
        )
        resolved = "dense" if is_dense else "short"
    else:
        resolved = requested
    return resolved, {
        "requested": requested,
        "resolved": resolved,
        "image_count": int(image_count),
        "motion_samples": len(samples),
        "motion_coverage": round(float(coverage), 4),
        "median_abs_dx": (None if median_abs_dx is None
                          else round(median_abs_dx, 4)),
    }
GENERAL_REUSE_REL_JUMP = _envf("GENERAL_REUSE_REL_JUMP", 0.22)
GENERAL_REUSE_ADJ_LOW_CONF = _envf("GENERAL_REUSE_ADJ_LOW_CONF", 0.25)
GENERAL_REUSE_ADJ_CONF_RATIO = _envf("GENERAL_REUSE_ADJ_CONF_RATIO", 0.30)
GENERAL_REUSE_ADJ_MOTION = _envf("GENERAL_REUSE_ADJ_MOTION", 0.85)
GENERAL_REUSE_ADJ_REL_JUMP = _envf("GENERAL_REUSE_ADJ_REL_JUMP", 0.10)
GENERAL_BREAK_MAX_TRACK = int(_envf("GENERAL_BREAK_MAX_TRACK", 3))
GENERAL_SINGLETON_SEP = _envf("GENERAL_SINGLETON_SEP", 0.14)
GENERAL_SINGLETON_STRONG_CONF = _envf("GENERAL_SINGLETON_STRONG_CONF", 0.75)
GENERAL_SINGLETON_PEER_CONF_RATIO = _envf("GENERAL_SINGLETON_PEER_CONF_RATIO", 0.85)
GENERAL_SINGLETON_CHAIN_MIN = int(_envf("GENERAL_SINGLETON_CHAIN_MIN", 3))
GENERAL_SINGLETON_CHAIN_APP = _envf("GENERAL_SINGLETON_CHAIN_APP", 0.60)
GENERAL_SINGLETON_PEAK_CONF = _envf("GENERAL_SINGLETON_PEAK_CONF", 0.50)
GENERAL_TEMPORAL_SINGLETON_CONF = _envf("GENERAL_TEMPORAL_SINGLETON_CONF", 0.70)
# 同一人被模型输出成两个相邻头框时，IoU 可能很低，普通 NMS 无法消除。
# 若中心距离小于较大框宽的该比例、纵向高度高度重合且尺度接近，保留高置信框。
GENERAL_DUP_CENTER = _envf("GENERAL_DUP_CENTER", 0.90)
GENERAL_DUP_Y_OVERLAP = _envf("GENERAL_DUP_Y_OVERLAP", 0.65)
GENERAL_DUP_SIZE_RATIO = _envf("GENERAL_DUP_SIZE_RATIO", 0.55)
GENERAL_DUP_APP = _envf("GENERAL_DUP_APP", 0.86)
GENERAL_DUP_TRACE = os.environ.get("GENERAL_DUP_TRACE", "") == "1"
GENERAL_COUNT_TRACE = os.environ.get("GENERAL_COUNT_TRACE", "") == "1"
GENERAL_PRETRACK_DEDUPE = os.environ.get("GENERAL_PRETRACK_DEDUPE", "") == "1"
GENERAL_ALIAS_MIN_FRAMES = int(_envf("GENERAL_ALIAS_MIN_FRAMES", 2))
GENERAL_ALIAS_MAX_FRAMES = int(_envf("GENERAL_ALIAS_MAX_FRAMES", 2))
GENERAL_ALIAS_APP = _envf("GENERAL_ALIAS_APP", 0.68)
GENERAL_ALIAS_CENTER = _envf("GENERAL_ALIAS_CENTER", 0.55)
GENERAL_ALIAS_SHADOW_IOU = _envf("GENERAL_ALIAS_SHADOW_IOU", 0.20)
GENERAL_ALIAS_SHADOW_CONF_RATIO = _envf("GENERAL_ALIAS_SHADOW_CONF_RATIO", 0.52)

# ---- 窗内 tracker 实现选择 ----
# 默认 windowhead = 移植版 WindowHeadTracker（见下方「★ 移植块」，已并入本文件），
#   实测 59 集 41/59 = 69.5%（MAE 0.356，多算 9 / 少算 9）。
# 设 WINHEAD_TRACKER=motion 可退回本文件自带的 MotionCoherentTracker
#   （59 集 35/59 = 59.3%），保留作对照。
TRACKER_IMPL = os.environ.get("WINHEAD_TRACKER", "windowhead").strip().lower()

# 窗过滤：窗框须被主车覆盖多少比例才算合法窗
WINDOW_COVER_MIN = 0.80
# 主车连续性：新主车中心离上一帧主车超过此像素则不切换（防跳到背景车）
CAR_SWITCH_MAX_DIST = 400.0


def imread_cn(path):
    """安全读取中文路径图片（cv2.imread 对中文路径返回 None）。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ==========================================================
# 2. 几何工具（与 nvidia_sideview_dualgate.py 保持一致）
# ==========================================================
def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def intersection_ratio(inner_box, outer_box):
    """inner_box 被 outer_box 覆盖的比例。"""
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = max(1.0, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
    return float(inter / inner_area)


def box_area(box):
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def suppress_near_duplicate_heads(dets, feats, confs, person_probs):
    """Remove a weak adjacent duplicate box without merging real neighbours.

    YOLO IoU NMS misses side-by-side duplicate head boxes whose overlap is
    modest.  Require near-identical vertical extent/size, centres within one
    head width, and a large confidence imbalance; crowded true heads with
    comparable confidence are preserved.
    """
    keep = [True] * len(dets)
    for i in range(len(dets)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(dets)):
            if not keep[j]:
                continue
            a, b = dets[i], dets[j]
            aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
            bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
            acx, acy = center(a)
            bcx, bcy = center(b)
            centre_distance = float(np.hypot(acx - bcx, acy - bcy))
            y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            y_ratio = y_overlap / max(1.0, min(ah, bh))
            x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            x_ratio = x_overlap / max(1.0, min(aw, bw))
            size_ratio = min(aw * ah, bw * bh) / max(1.0, max(aw * ah, bw * bh))
            confidence_ratio = min(float(confs[i]), float(confs[j])) / max(
                1e-6, float(confs[i]), float(confs[j]))
            appearance_sim = None
            if feats[i] is not None and feats[j] is not None:
                appearance_sim = float(np.clip(
                    np.dot(feats[i], feats[j]), -1.0, 1.0))
            if (centre_distance <= 0.80 * min(aw, bw) and
                    y_ratio >= 0.82 and size_ratio >= 0.65 and
                    confidence_ratio <= 0.55 and
                    (x_ratio >= 0.55 or
                     (appearance_sim is not None and appearance_sim >= 0.90))):
                loser = i if float(confs[i]) < float(confs[j]) else j
                keep[loser] = False
                if GENERAL_COUNT_TRACE:
                    print(f"[邻近重复头框过滤] drop=#{loser + 1} "
                          f"conf={float(confs[loser]):.3f} "
                          f"pair=({i + 1},{j + 1}) dist={centre_distance:.1f} "
                          f"x={x_ratio:.2f} y={y_ratio:.2f} size={size_ratio:.2f} "
                          f"app={appearance_sim if appearance_sim is not None else -1:.2f} "
                          f"conf_ratio={confidence_ratio:.2f}")
                if loser == i:
                    break
    indices = [idx for idx, value in enumerate(keep) if value]
    return ([dets[idx] for idx in indices],
            [feats[idx] for idx in indices],
            [confs[idx] for idx in indices],
            [person_probs[idx] for idx in indices])



# ==========================================================
# 3. 外观特征（OSNet，全局共享一个实例）
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

_reid_model = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_reid_model():
    global _reid_model
    if _reid_model is None:
        _reid_model = models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        ).to(device).eval()
        print("[OSNet外观特征] 已加载 torchreid 预训练 OSNet_x1_0")
    return _reid_model


def extract_feat(img_crop):
    """把一个人头裁剪图变成 L2 归一化的外观特征向量（OSNet）。"""
    img = _transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = _get_reid_model()(img)
    feat = feat.cpu().numpy().flatten()
    return feat / (np.linalg.norm(feat) + 1e-6)


def extract_feat_and_person_prob(img_crop):
    """仅使用原始 OSNet x1.0 提取 ReID 特征，不进行真假人分类。"""
    tensor = _transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        reid_feat = _get_reid_model()(tensor)
        feat = reid_feat.cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-6)
    # None 表示没有真假人分类分数；分类阈值过滤逻辑因此保持关闭。
    return feat, None, None


# ==========================================================
# 4. WindowIdTracker —— 跨帧稳定的车窗编号
#    （照搬 nvidia_sideview_dualgate.py 514–757）
# ==========================================================
class WindowIdTracker:
    """给每个物理车窗分配跨帧稳定的编号，用于逐窗隔离跟踪。

    首帧：按 x 中心排序编号（方向由 `is_car_front_on_left` 决定 motion_sign）。
    后续：用车框位移做运动补偿 + 代价匹配延续编号；匹配不上的给 TTL 保留。
    """

    def __init__(self, max_missed_ttl=4):
        self.previous_windows = {}     # {wid: {'box','area','missed'}}
        self.previous_car = None
        self.last_motion = (0.0, 0.0)
        self.last_motion_clipped_fallback = False
        self.max_missed_ttl = max_missed_ttl
        self.next_window_id = 1
        self.motion_sign = None
        self.cum_dx = 0.0              # 车框累计横向位移，用于判定行驶方向
        self.motion_dx_history = []    # 有窗帧的逐帧位移，供序列末尾鲁棒确认方向
        # 序列末尾是否允许整体反转，使用相互独立的车辆框和多车窗运动证据。
        # 任一检测器都可能发生框扩张/收缩或中心跳变，因此不再允许单一
        # 检测框来源独自触发 W1..Wn 的整体反转。
        self.direction_evidence = []   # [(frame_source, dx, weight), ...]

    def _estimate_shift(self, car_box, img_width):
        """整车帧间位移；车框被图像边缘裁切时用另一侧边界对齐。"""
        dx, dy = 0.0, 0.0
        if car_box is not None and self.previous_car is not None:
            left_cropped = (car_box[0] <= 15.0)
            right_cropped = (car_box[2] >= img_width - 15.0)
            if left_cropped and not right_cropped:
                dx = car_box[2] - self.previous_car[2]
                dy = car_box[3] - self.previous_car[3]
            elif right_cropped and not left_cropped:
                dx = car_box[0] - self.previous_car[0]
                dy = car_box[1] - self.previous_car[1]
            else:
                cx_c, cy_c = center(car_box)
                cx_p, cy_p = center(self.previous_car)
                dx, dy = cx_c - cx_p, cy_c - cy_p
        elif self.last_motion is not None:
            dx, dy = self.last_motion
        return dx, dy

    def update(self, windows, car_box, img_width, is_car_front_on_left=True):
        self.last_motion_clipped_fallback = False
        # ---- 本帧没有窗检测：只做运动外推 + TTL 保留 ----
        if not windows:
            self.cum_dx += self.last_motion[0]
            kept = {}
            dx, dy = self.last_motion
            for wid, p_info in self.previous_windows.items():
                missed_cnt = p_info.get('missed', 0) + 1
                pred_box = [p_info['box'][0] + dx, p_info['box'][1] + dy,
                            p_info['box'][2] + dx, p_info['box'][3] + dy]
                pred_cx = (pred_box[0] + pred_box[2]) / 2.0
                max_ttl = self.max_missed_ttl + (2 if wid == 1 else 0)
                if missed_cnt <= max_ttl and -50.0 <= pred_cx <= img_width + 50.0:
                    kept[wid] = {'box': pred_box, 'area': p_info.get('area', 0.0),
                                 'missed': missed_cnt}
            self.previous_windows = kept
            if car_box is not None:
                self.previous_car = car_box
            return []

        dx, dy = self._estimate_shift(car_box, img_width)
        raw_car_dx = float(dx)
        previous_car_for_direction = self.previous_car
        # 车辆框在裁切边缘可能突然扩张/收缩，中心或单侧边界因此给出
        # 反向、超大的伪位移。利用已有窗口与当前窗口之间、沿已确认行驶
        # 方向的成对位移取中位数兜底；至少两个窗口支持才采用，避免单窗
        # 自身检测抖动改变运动估计。
        if self.motion_sign is not None and len(self.previous_windows) >= 2 and len(windows) >= 2:
            directional_dx = []
            for p_info in self.previous_windows.values():
                pcx = center(p_info['box'])[0]
                for curr_box in windows:
                    delta = float(center(curr_box)[0] - pcx)
                    if delta * float(self.motion_sign) > 2.0 and abs(delta) <= 350.0:
                        directional_dx.append(delta)
            car_motion_bad = (
                dx * float(self.motion_sign) <= 0.0 or
                abs(float(dx)) >= 350.0)
            # When the vehicle is clipped by an image edge, its box expands or
            # contracts instead of translating rigidly.  The retained edge can
            # therefore under-estimate motion even with the correct sign.  Use
            # rank-aligned windows (left-to-left, middle-to-middle, ...) as a
            # consensus check; require at least two windows and a large enough
            # disagreement so ordinary detector jitter cannot change tracking.
            prev_ranked = sorted(
                (info['box'] for info in self.previous_windows.values()),
                key=lambda b: center(b)[0])
            curr_ranked = sorted(windows, key=lambda b: center(b)[0])
            aligned_dx = [float(center(cb)[0] - center(pb)[0])
                          for pb, cb in zip(prev_ranked, curr_ranked)]
            aligned_dx = [v for v in aligned_dx
                          if v * float(self.motion_sign) > 2.0 and abs(v) <= 350.0]
            # Only the leading/exiting edge is unreliable.  At the opposite
            # (entry) edge _estimate_shift deliberately follows the visible
            # boundary and is usually more reliable than rank alignment while
            # new windows are still appearing.
            car_clipped = (
                car_box is not None and
                ((self.motion_sign < 0 and car_box[0] <= 15.0) or
                 (self.motion_sign > 0 and car_box[2] >= img_width - 15.0)))
            aligned_window_dx = (float(np.median(aligned_dx))
                                 if (len(aligned_dx) >= 2 and
                                     len(prev_ranked) == len(curr_ranked))
                                 else None)
            clipped_motion_bad = (
                car_clipped and aligned_window_dx is not None and
                abs(aligned_window_dx - float(dx)) >= 70.0)
            # Preserve the established bad-sign/huge-motion fallback exactly;
            # it is also used by historical edge-exit sequences.  The new
            # clipped-box correction is only for otherwise plausible car
            # motion that under-estimates the window consensus.
            if car_motion_bad and len(directional_dx) >= 2:
                window_dx = float(np.median(directional_dx))
                clipped_motion_bad = False
            elif clipped_motion_bad:
                window_dx = aligned_window_dx
                self.last_motion_clipped_fallback = True
            else:
                window_dx = None
            if window_dx is not None:
                if GENERAL_COUNT_TRACE:
                    print(f"[车窗运动回退] car_dx={dx:.1f} -> window_dx={window_dx:.1f} "
                          f"sign={self.motion_sign} samples="
                          f"{len(aligned_dx) if clipped_motion_bad else len(directional_dx)}")
                dx = window_dx
        self.last_motion = (dx, dy)
        self.cum_dx += dx
        if abs(float(dx)) > 2.0:
            self.motion_dx_history.append(float(dx))

        # ---- 仅供序列末尾方向确认的可信证据 ----
        # 1) 车辆框证据：宽高不能突变，位移不能是离群大跳。
        if car_box is not None and previous_car_for_direction is not None:
            prev_w = max(1.0, float(previous_car_for_direction[2] - previous_car_for_direction[0]))
            prev_h = max(1.0, float(previous_car_for_direction[3] - previous_car_for_direction[1]))
            curr_w = max(1.0, float(car_box[2] - car_box[0]))
            curr_h = max(1.0, float(car_box[3] - car_box[1]))
            width_ratio = curr_w / prev_w
            height_ratio = curr_h / prev_h
            car_geometry_stable = (
                0.75 <= width_ratio <= 1.33 and
                0.75 <= height_ratio <= 1.33 and
                2.0 < abs(raw_car_dx) <= 350.0)
            if car_geometry_stable:
                self.direction_evidence.append(("car", raw_car_dx, 2.0))

        # 2) 多车窗共识证据：只使用前后数量相同、至少两个窗口时的
        # 排名对齐位移；同时要求窗口尺寸稳定、方向高度一致且离散度小。
        # 单窗、漏窗/增窗以及框突变帧均不能提供最终方向证据。
        if len(self.previous_windows) >= 2 and len(windows) == len(self.previous_windows):
            prev_ranked_dir = sorted(
                (info['box'] for info in self.previous_windows.values()),
                key=lambda b: center(b)[0])
            curr_ranked_dir = sorted(windows, key=lambda b: center(b)[0])
            window_dxs = []
            geometry_ok = True
            for prev_box, curr_box in zip(prev_ranked_dir, curr_ranked_dir):
                pw = max(1.0, float(prev_box[2] - prev_box[0]))
                ph = max(1.0, float(prev_box[3] - prev_box[1]))
                cw = max(1.0, float(curr_box[2] - curr_box[0]))
                ch = max(1.0, float(curr_box[3] - curr_box[1]))
                if not (0.65 <= cw / pw <= 1.55 and 0.65 <= ch / ph <= 1.55):
                    geometry_ok = False
                    break
                window_dxs.append(float(center(curr_box)[0] - center(prev_box)[0]))
            if geometry_ok and len(window_dxs) >= 2:
                window_median_dx = float(np.median(window_dxs))
                same_direction = sum(
                    1 for value in window_dxs
                    if value * window_median_dx > 0.0 and abs(value) > 2.0)
                direction_ratio = same_direction / len(window_dxs)
                mad = float(np.median(np.abs(
                    np.asarray(window_dxs, dtype=np.float32) - window_median_dx)))
                if (2.0 < abs(window_median_dx) <= 350.0 and
                        direction_ratio >= 0.75 and
                        mad <= max(20.0, 0.50 * abs(window_median_dx) + 10.0)):
                    self.direction_evidence.append(("windows", window_median_dx, 3.0))

        # ---- 编号朝向 ----
        # 规则（用户 2026-10-09 指定）：车在画面里从左往右行驶 ⇒ 最右的窗是 W1，
        # 次右是 W2，依此类推；从右往左行驶则镜像（最左 = W1）。即 W1 恒为
        # **车头方向的领头窗**。行驶方向用车框累计位移 cum_dx 判定（比单帧 dx 稳）。
        # ⚠ 原实现在 update() 第一行就锁死，而第一帧常常一个窗都没存活
        #   （全是背景车或被覆盖率过滤）→ is_car_front_on_left 只是函数默认值 True，
        #   方向被永久判反。实测桌面序列 20260819190824959：真值是往右开，
        #   却被锁成 motion_sign=-1（W1=最左），后续「顺序约束」再把它强行对调，
        #   造成帧 3→4 的 W1/W2 互换。
        # ⇒ 改为**只在「真的有窗可编号」的那一帧锁定**。
        if self.motion_sign is None:
            if abs(self.cum_dx) >= _HU_TRAVEL_MIN_DX:
                travel_right = self.cum_dx > 0.0
            else:
                travel_right = not is_car_front_on_left
            self.motion_sign = 1 if travel_right else -1
        elif (len(self.previous_windows) <= 1
              and abs(self.cum_dx) >= _HU_TRAVEL_MIN_DX):
            # 首个车窗出现时车框尚无足够位移，方向只能是几何猜测。
            # 一旦累计位移足够，且此前最多只有一个物理窗，只更正
            # 后续的窗口排序方向，不改已有 window/head tracker ID。
            # 已有两个及以上窗时不再改方向，避免交换已积累的人头历史。
            confirmed_motion_sign = 1 if self.cum_dx > 0.0 else -1
            if confirmed_motion_sign != self.motion_sign:
                self.motion_sign = confirmed_motion_sign

        # ---- 首帧：按 x 中心排序编号 ----
        if not self.previous_windows:
            sorted_curr = sorted(
                enumerate(windows),
                key=lambda x: -self.motion_sign * ((x[1][0] + x[1][2]) / 2.0))
            assigned = {curr_idx: rank + 1 for rank, (curr_idx, _) in enumerate(sorted_curr)}
            result = [(assigned[i], windows[i]) for i in range(len(windows))]
            self.previous_windows = {
                wid: {'box': box, 'area': box_area(box), 'missed': 0}
                for wid, box in result}
            self.next_window_id = max(self.previous_windows.keys(), default=0) + 1
            self.previous_car = car_box
            return result

        curr_sorted = sorted(
            enumerate(windows),
            key=lambda x: -self.motion_sign * ((x[1][0] + x[1][2]) / 2.0))
        assigned = {}
        matched_prev_ids = set()

        # 1 号窗（前排领头窗）快出画时不再被匹配，避免抢到后面新出现的窗
        w1_exit_pending = False
        if 1 in self.previous_windows:
            w1_cx = center(self.previous_windows[1]['box'])[0]
            w1_exit_pending = (w1_cx <= 80.0) if self.motion_sign < 0 \
                else (w1_cx >= img_width - 80.0)
            # 车头已被画面领先边缘裁切时，W1 可能在其框中心还没有
            # 走到画外前就完全消失；同时车尾会从另一侧新露出一个
            # 较窄的后续窗。这种“领先窗退出 + 尾部窗进入”不能按当前
            # 可见窗重新从 1 编号，应保留 W2/W3 并给新窗新 ID。
            if (not w1_exit_pending and car_box is not None and
                    len(windows) >= len(self.previous_windows) and len(windows) >= 2):
                car_w = max(1.0, float(car_box[2] - car_box[0]))
                leading_cropped = (
                    (self.motion_sign > 0 and car_box[2] >= img_width - 30.0) or
                    (self.motion_sign < 0 and car_box[0] <= 30.0))
                trailing_idx = min(
                    range(len(windows)),
                    key=lambda i: self.motion_sign * center(windows[i])[0])
                trailing_box = windows[trailing_idx]
                trailing_w = max(1.0, float(trailing_box[2] - trailing_box[0]))
                other_widths = [
                    max(1.0, float(b[2] - b[0]))
                    for i, b in enumerate(windows) if i != trailing_idx]
                w1_prev_box = self.previous_windows[1]['box']
                w1_prev_w = max(1.0, float(w1_prev_box[2] - w1_prev_box[0]))
                w1_pred_cx = center(w1_prev_box)[0] + dx
                # 边缘裁切只是“可能退出”。若当前仍有与 W1 运动预测
                # 紧密吻合、宽度相仿的窗口，说明 W1 尚未消失，禁止提前
                # 把它释放并改成 W3（实测车辆 106）。
                w1_continuation_visible = any(
                    abs(center(b)[0] - w1_pred_cx) <=
                    0.30 * max(w1_prev_w, float(b[2] - b[0]), 1.0)
                    and min(w1_prev_w, max(1.0, float(b[2] - b[0]))) /
                    max(w1_prev_w, max(1.0, float(b[2] - b[0]))) >= 0.55
                    for b in windows)
                trailing_pos = ((center(trailing_box)[0] - car_box[0]) / car_w
                                if self.motion_sign > 0 else
                                (car_box[2] - center(trailing_box)[0]) / car_w)
                narrow_trailing_entry = (
                    other_widths and
                    trailing_w <= 0.45 * float(np.median(other_widths)) and
                    trailing_pos <= 0.22)
                if (leading_cropped and narrow_trailing_entry and
                        not w1_continuation_visible):
                    w1_exit_pending = True

        # ---- 代价匹配 ----
        # A large inter-frame motion can move the leading pane completely out
        # before its old centre reaches the image edge. Detect that before cost
        # matching so predicted W1 cannot steal physical W2.
        if (not w1_exit_pending and 1 in self.previous_windows and
                car_box is not None and windows):
            leading_cropped = (
                (self.motion_sign > 0 and car_box[2] >= img_width - 30.0) or
                (self.motion_sign < 0 and car_box[0] <= 30.0))
            w1_info = self.previous_windows[1]
            w1_box = w1_info['box']
            w1_missed = int(w1_info.get('missed', 0)) + 1
            w1_pred_cx = center(w1_box)[0] + dx * w1_missed
            w1_width = max(1.0, float(w1_box[2] - w1_box[0]))
            current_centres = [center(b)[0] for b in windows]
            leading_current = (max(current_centres) if self.motion_sign > 0
                               else min(current_centres))
            predicted_ahead = self.motion_sign * (w1_pred_cx - leading_current)
            if (leading_cropped and len(self.previous_windows) >= 2 and
                    predicted_ahead > max(90.0, 0.55 * w1_width)):
                w1_exit_pending = True
                if GENERAL_COUNT_TRACE:
                    print(f"[W1 predicted exit] pred={w1_pred_cx:.1f} "
                          f"lead={leading_current:.1f} "
                          f"ahead={predicted_ahead:.1f} width={w1_width:.1f}")

        match_candidates = []
        for curr_idx, curr_box in enumerate(windows):
            curr_cx, curr_cy = center(curr_box)
            curr_w = max(1.0, curr_box[2] - curr_box[0])
            curr_h = max(1.0, curr_box[3] - curr_box[1])
            curr_area = box_area(curr_box)
            for prev_id, p_info in self.previous_windows.items():
                if prev_id == 1 and w1_exit_pending:
                    continue
                prev_box = p_info['box']
                prev_area = p_info.get('area', box_area(prev_box))
                prev_w = max(1.0, prev_box[2] - prev_box[0])
                prev_h = max(1.0, prev_box[3] - prev_box[1])
                missed_k = p_info.get('missed', 0) + 1
                pred_cx = center(prev_box)[0] + dx * missed_k
                pred_cy = center(prev_box)[1] + dy * missed_k
                dist_x = abs(pred_cx - curr_cx)
                dist_y = abs(pred_cy - curr_cy)
                area_sim = min(curr_area, prev_area) / max(curr_area, prev_area)
                width_sim = min(curr_w, prev_w) / max(curr_w, prev_w)
                height_sim = min(curr_h, prev_h) / max(curr_h, prev_h)
                if dist_x < 260.0 and dist_y < 120.0 and area_sim >= 0.20:
                    cost = (dist_x + 0.5 * dist_y
                            + 180.0 * (1.0 - area_sim)
                            + 60.0 * (1.0 - width_sim)
                            + 40.0 * (1.0 - height_sim)
                            + 120.0 * p_info.get('missed', 0))
                    match_candidates.append((cost, curr_idx, prev_id))

        for cost, curr_idx, prev_id in sorted(match_candidates, key=lambda x: x[0]):
            if curr_idx in assigned or prev_id in matched_prev_ids:
                continue
            assigned[curr_idx] = prev_id
            matched_prev_ids.add(prev_id)

        # ---- 1 号窗强制延续（领头窗最稳，单独保一手）----
        if 1 in self.previous_windows and 1 not in matched_prev_ids and not w1_exit_pending:
            w1_box = self.previous_windows[1]['box']
            w1_missed = self.previous_windows[1].get('missed', 0) + 1
            w1_pred_cx = center(w1_box)[0] + dx * w1_missed
            unassigned = [i for i in range(len(windows)) if i not in assigned]
            if unassigned:
                best_idx = min(unassigned,
                               key=lambda i: abs(center(windows[i])[0] - w1_pred_cx))
                w1_residual = abs(center(windows[best_idx])[0] - w1_pred_cx)
                w1_width = max(1.0, float(w1_box[2] - w1_box[0]))
                # W1 used to be forced onto the nearest remaining pane with no
                # distance limit.  Once the leading pane has left the image,
                # that steals physical W2 and shifts W2/W3 to W1/W2.  Preserve
                # the special W1 continuity only while its prediction remains
                # geometrically plausible; otherwise TTL/fallback can retain
                # the true suffix IDs.
                if w1_residual <= max(90.0, 0.65 * w1_width):
                    assigned[best_idx] = 1
                    matched_prev_ids.add(1)
                elif GENERAL_COUNT_TRACE:
                    print(f"[W1强制延续抑制] pred={w1_pred_cx:.1f} "
                          f"det={center(windows[best_idx])[0]:.1f} "
                          f"resid={w1_residual:.1f} width={w1_width:.1f}")

        # ---- 窗数变少时按顺序补配对 ----
        fallback_curr = set()
        if len(windows) <= len(self.previous_windows):
            remaining_prev = [wid for wid in sorted(self.previous_windows)
                              if wid not in matched_prev_ids
                              and not (wid == 1 and w1_exit_pending)]
            remaining_curr = [i for i, _ in curr_sorted if i not in assigned]
            for curr_idx, wid in zip(remaining_curr, remaining_prev):
                assigned[curr_idx] = wid
                matched_prev_ids.add(wid)
                fallback_curr.add(curr_idx)

        # ---- 剩下的窗按车身方向发新编号 ----
        # 检测器的返回顺序不等于物理窗口顺序；同帧新出现多个窗时
        # 必须沿 W1 -> Wn 的方向递增，否则会出现 W4,W3,W1,W2。
        # A clipped remainder of the leading W1 can survive for a few frames
        # after W1 has entered the exit-pending state. Preserve W1 for that
        # edge fragment instead of allocating a false new W3. A genuinely new
        # window enters from the opposite edge and is unaffected.
        if w1_exit_pending and 1 not in matched_prev_ids:
            unassigned = [i for i in range(len(windows)) if i not in assigned]
            matched_widths = [
                max(1.0, float(windows[i][2] - windows[i][0]))
                for i in assigned
            ]
            exit_candidates = []
            for i in unassigned:
                b = windows[i]
                touches_exit = ((b[0] <= 12.0) if self.motion_sign < 0
                                else (b[2] >= img_width - 12.0))
                bw = max(1.0, float(b[2] - b[0]))
                narrow_residual = (not matched_widths or
                                   bw <= 0.65 * float(np.median(matched_widths)))
                if touches_exit and narrow_residual:
                    exit_candidates.append(i)
            if exit_candidates:
                if self.motion_sign < 0:
                    keep_idx = min(exit_candidates,
                                   key=lambda i: center(windows[i])[0])
                else:
                    keep_idx = max(exit_candidates,
                                   key=lambda i: center(windows[i])[0])
                assigned[keep_idx] = 1
                matched_prev_ids.add(1)

        for curr_idx, _ in curr_sorted:
            if curr_idx not in assigned:
                assigned[curr_idx] = self.next_window_id
                self.next_window_id += 1

        result = [(assigned[i], windows[i]) for i in range(len(windows))]

        if GENERAL_COUNT_TRACE:
            _fmt_box = lambda b: tuple(round(float(v), 1) for v in b)
            print(f"[窗口匹配] dx={dx:.1f} sign={self.motion_sign} "
                  f"car={_fmt_box(car_box) if car_box is not None else None} "
                  f"prev={[(wid, _fmt_box(info['box'])) for wid, info in self.previous_windows.items()]} "
                  f"curr={[(wid, _fmt_box(box)) for wid, box in result]}")

        # ---- 顺序约束：按行驶方向，窗号必须单调（W1 恒在最前）----
        # 车头方向固定 ⇒ 各窗沿车轴的左右次序固定 ⇒ 编号次序也必须固定。
        # 代价匹配失败后走的"按顺序补配对"是按编号 zip 的，可能产生左右逆序
        # （实测 20260714180408088 的 W1/W2 互换即源于此）。
        # 这里只对**继承自上一帧的编号**做一次排列修正：不新增/不删除编号，
        # 且不动本帧新开的编号（免得刚出现的新窗抢走 W1）。
        # HU_WIN_ORDER_FIX 限定修正范围（默认 all = 上游行为）：
        #   all      —— 修正全部继承编号（上游行为）。⚠ 会覆盖【代价匹配已经配对的】
        #               结果：当新窗开在旧窗“错误的一侧”（如 W1 在右、新窗 W2 在左）
        #               时，本帧两窗都是继承编号 → 这里把两者强行对调 = W1/W2 互换，
        #               跨窗 ID 空间分裂 → 同一个人被算两次（实测 20260819190824959
        #               帧 4：代价匹配残差 +34.5/+3.5，被修正后变成 -170.5/+208.5）。
        #   fallback —— 只修正「按顺序补配对」那一支（第 2 号偏离的原始目标），
        #               不干涉代价匹配的结果。
        #   off      —— 完全关闭。
        if _HU_WIN_ORDER_FIX == "fallback":
            pool = [i for i in range(len(result)) if i in fallback_curr]
        elif _HU_WIN_ORDER_FIX == "off":
            pool = []
        else:
            pool = [i for i in range(len(result)) if result[i][0] in matched_prev_ids]
        order = sorted(pool,
                       key=lambda i: -self.motion_sign * center(result[i][1])[0])
        wids_sorted = sorted(result[i][0] for i in order)
        for pos, i in enumerate(order):
            result[i] = (wids_sorted[pos], result[i][1])

        # ---- 维护 previous_windows（含 TTL）----
        new_previous = {wid: {'box': box, 'area': box_area(box), 'missed': 0}
                        for wid, box in result}
        for prev_id, p_info in self.previous_windows.items():
            if prev_id not in matched_prev_ids:
                missed_cnt = p_info.get('missed', 0) + 1
                pb = p_info['box']
                pred_box = [pb[0] + dx, pb[1] + dy, pb[2] + dx, pb[3] + dy]
                pred_cx = (pred_box[0] + pred_box[2]) / 2.0
                if missed_cnt <= self.max_missed_ttl and -50.0 <= pred_cx <= img_width + 50.0:
                    new_previous[prev_id] = {'box': pred_box,
                                             'area': p_info.get('area', 0),
                                             'missed': missed_cnt}
        self.previous_windows = new_previous
        if car_box is not None:
            self.previous_car = car_box
        return result


# ==========================================================
# 5. MotionCoherentTracker —— 窗内 head 跟踪（单窗一个实例）
#    改动：update() 新增 motion_shift 入参；去掉 M-A/M-B/M-D 与外观开关。
# ==========================================================
class HeadTrack:
    def __init__(self, global_id, bbox, feat):
        self.global_id = global_id
        self.bbox = bbox
        self.feat_history = deque(maxlen=5)
        self.feat_history.append(feat)
        self.age = 0
        self.velocity = np.array([0.0, 0.0])
        self.missed_shift = np.array([0.0, 0.0])


class MotionCoherentTracker:
    """单窗内的协同运动跟踪器。

    运动补偿来源（优先级）：
      1. update() 传入的 `motion_shift`（整车车框位移，正常路径）
      2. 自行 RANSAC 估计（车框丢失时兜底；需 >=2 个点对）
    """

    def __init__(self):
        self.tracks = []
        self.next_global_id = 0
        self.global_id_history = {}          # {gid: deque([feat], maxlen=5)}
        self.global_vehicle_velocity = np.array([0.0, 0.0])
        self.global_shift = np.array([0.0, 0.0])
        self.shift_inliers = 0
        self.shift_source = "none"
        self.frame_idx = 0

    def _estimate_global_shift(self, dets, tol=RANSAC_TOL):
        """RANSAC 式整车帧间平移估计（兜底用；需 >=2 个 (轨迹,检测) 点对）。"""
        track_c = [np.array(center(t.bbox)) for t in self.tracks]
        det_c = [np.array(center(d)) for d in dets]
        if not track_c or not det_c:
            return np.array([0.0, 0.0]), 0
        best_shift, best_n = np.array([0.0, 0.0]), 0
        for tc in track_c:
            for dc in det_c:
                s = dc - tc
                n = sum(1 for tc2 in track_c for dc2 in det_c
                        if np.linalg.norm(dc2 - tc2 - s) < tol)
                if n > best_n:
                    best_n, best_shift = n, s
        return best_shift, best_n

    def update(self, dets, feats, confs, motion_shift=None):
        self.frame_idx += 1
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()

        if len(dets) == 0:
            return []

        # ---- 运动补偿：优先用传入的整车 shift，否则回退 RANSAC ----
        if motion_shift is not None:
            self.global_shift = np.array(motion_shift, dtype=float)
            self.shift_inliers = -1
            self.shift_source = "car"
            use_shift = True
        else:
            self.global_shift, self.shift_inliers = self._estimate_global_shift(dets)
            self.shift_source = "ransac"
            use_shift = self.shift_inliers >= 2

        predicted_centers = {}
        for j, t in enumerate(self.tracks):
            cx, cy = center(t.bbox)
            if use_shift:
                predicted_centers[j] = np.array([cx, cy]) + self.global_shift + t.missed_shift
            else:
                v = t.velocity if np.linalg.norm(t.velocity) > 0 else self.global_vehicle_velocity
                predicted_centers[j] = np.array([cx, cy]) + v * t.age

        dist_gate = DIST_GATE if use_shift else 150.0

        # OSD-only identity handoff: if the nearest mature whole-car track has
        # a large residual even after vehicle-motion compensation and its best
        # appearance similarity is low, an adjacent-frame detection is a new
        # passenger rather than the old global ID. Per-window counting is not
        # changed by this guard.
        forced_new_dets = set()
        if use_shift and len(dets) >= 2 and self.tracks:
            for i, det in enumerate(dets):
                dc = np.array(center(det))
                nearest_j = min(predicted_centers,
                                key=lambda j: np.linalg.norm(dc - predicted_centers[j]))
                nearest_t = self.tracks[nearest_j]
                residual = float(np.linalg.norm(dc - predicted_centers[nearest_j]))
                sims = np.dot(np.array(nearest_t.feat_history), feats[i])
                best_sim = float(np.max(sims)) if len(sims) else 1.0
                if (nearest_t.age == 1 and len(nearest_t.feat_history) >= 3 and
                        (residual >= 60.0 or
                         (residual >= 50.0 and best_sim < 0.65))):
                    forced_new_dets.add(i)

        # ---- 阶段一：运动补偿位置 + 外观 加权匹配 ----
        matches = []
        for i, det in enumerate(dets):
            if i in forced_new_dets:
                continue
            d_cx, d_cy = center(det)
            c = confs[i]
            s_c = (c * c) / (c * c + (1.0 - c) * (1.0 - c))
            w_m = W_MOT_BASE + W_MOT_RANGE * s_c       # 运动权重限 [0.2,0.5]，外观主导
            for j, t in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                dist = np.linalg.norm(np.array([d_cx, d_cy]) - predicted_centers[j])
                if dist >= dist_gate:
                    continue
                sims = np.dot(np.array(t.feat_history), feats[i])
                sim = float(np.max(sims))
                if sim <= SIM_GATE:
                    continue
                motion_score = 1.0 - dist / dist_gate
                matches.append((w_m * motion_score + (1.0 - w_m) * sim, i, j))

        matches.sort(key=lambda x: x[0], reverse=True)
        velocity_samples = []
        for score, i, j in matches:
            if assigned_indices[i] != -1 or j in matched_tracks:
                continue
            assigned_indices[i] = j
            matched_tracks.add(j)
            t = self.tracks[j]
            old_cx, old_cy = center(t.bbox)
            new_cx, new_cy = center(dets[i])
            inst_velocity = np.array([new_cx - old_cx, new_cy - old_cy])
            w = float(np.clip(confs[i], 0.2, 0.9))
            t.velocity = t.velocity * (1.0 - w) + inst_velocity * w
            velocity_samples.append(t.velocity)
            t.bbox = dets[i]
            t.feat_history.append(feats[i])
            t.age = 0
            t.missed_shift = np.array([0.0, 0.0])

        if velocity_samples:
            self.global_vehicle_velocity = np.mean(velocity_samples, axis=0)

        # ---- 阶段二：未匹配检测 → 回溯认领旧身份 ----
        claimed_gids = {self.tracks[j].global_id for j in matched_tracks}
        pre_track_indices = list(range(len(self.tracks)))

        for i in range(len(dets)):
            if assigned_indices[i] != -1:
                continue
            det_feat = feats[i]
            d_cx, d_cy = center(dets[i])
            matched_gid = -1
            best_sim = 0.0

            # 置信度自适应认领门限：低置信检测需要更高外观相似度
            claim_thr = CLAIM_THR + (1.0 - confs[i]) * CLAIM_RANGE
            for gid, feat_list in (() if i in forced_new_dets
                                   else self.global_id_history.items()):
                if gid in claimed_gids or not feat_list:
                    continue
                sim = float(np.max(np.dot(np.array(feat_list), det_feat)))
                if sim > best_sim:
                    best_sim = sim
                    if sim > claim_thr:
                        matched_gid = gid

            # 空间兜底：外观失败时，用预测位置最近的未匹配轨迹
            spatial_j = -1
            if matched_gid == -1 and i not in forced_new_dets:
                best_d = SPATIAL_R * (0.5 + 0.5 * confs[i])
                for j in pre_track_indices:
                    if j in matched_tracks:
                        continue
                    if self.tracks[j].global_id in claimed_gids:
                        continue
                    dist = np.linalg.norm(np.array([d_cx, d_cy]) - predicted_centers[j])
                    if dist < best_d:
                        best_d, spatial_j = dist, j
                if spatial_j != -1:
                    matched_gid = self.tracks[spatial_j].global_id
                    matched_tracks.add(spatial_j)
                    best_sim = 0.0

            if matched_gid == -1:
                matched_gid = self.next_global_id
                self.next_global_id += 1
                self.global_id_history[matched_gid] = deque([det_feat], maxlen=5)
            else:
                self.global_id_history[matched_gid].append(det_feat)
            claimed_gids.add(matched_gid)

            # 复用同 gid 的存活轨迹，否则新建
            existing_idx = next((idx for idx, t in enumerate(self.tracks)
                                 if t.global_id == matched_gid), None)
            if existing_idx is not None:
                t = self.tracks[existing_idx]
                t.bbox = dets[i]
                t.feat_history.append(det_feat)
                t.age = 0
                t.missed_shift = np.array([0.0, 0.0])
                assigned_indices[i] = existing_idx
            else:
                self.tracks.append(HeadTrack(matched_gid, dets[i], det_feat))
                assigned_indices[i] = len(self.tracks) - 1

        # ---- 未被匹配的旧轨迹：累计整车位移供下次预测 ----
        acc_shift = self.global_shift if use_shift else self.global_vehicle_velocity
        for j in pre_track_indices:
            if j not in matched_tracks:
                self.tracks[j].missed_shift += acc_shift

        # assigned_indices refers to the pre-pruning list.  On long/dense
        # sequences an expired track can be removed from the middle, shifting
        # every later index and causing either a wrong ID or IndexError.  Freeze
        # the IDs before compacting the live-track list.
        assigned_gids = [self.tracks[idx].global_id for idx in assigned_indices]
        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]

        return assigned_gids

    def total_count(self):
        """该窗跨帧认出的不同身份数（distinct）。"""
        return len(self.global_id_history)


# ==========================================================================
# ★ 移植块开始：以下系 nvidia_distinct_noguard.py 第 972-1917 行的**逐行移植**
#   （class WindowHeadTracker 的跟踪核心：__init__ / _archive_appearance /
#     _max_sim / _claim_expired_identity / _mark_coexisting_ids / _cannot_merge /
#     update，外加 count_distinct）。
#
#   **勿手改** —— 改了就不再是"移植版"，与源脚本的 re-diff 会失效。
#   需要重新生成时运行实验室脚本目录下的 _port_wht.py。
#   计数口径：count_distinct() = len(track_summaries)，裸 distinct，无座位/容量先验。
# ==========================================================================
# ---- 模块级参数（原文件全局；默认值 = 原脚本运行时实际生效值）----
_MATCH_MODE = os.environ.get("WHT_MATCH_MODE", "sidegate")   # sidegate / hybrid / position
_APP_GATE = float(os.environ.get("WHT_APP_GATE", 0.55))      # 外观硬 AND 门
_APP_HIST = int(os.environ.get("WHT_APP_HIST", 5))           # 每 gid 保留的外观帧数
_CLAIM_THR = float(os.environ.get("WHT_CLAIM_THR", 0.65))    # 阶段二全局认领门限（原 0.78，见顶部说明第 3 条）
_WM_MOT = float(os.environ.get("WHT_WM_MOT", 0.35))          # sidegate 运动权重
_HYBRID_WM = float(os.environ.get("WHT_HYBRID_WM", 0.5))     # hybrid 模式运动权重
_ENABLE_CLAIM = os.environ.get("WHT_ENABLE_CLAIM", "1") != "0"
# R-1：V7 历史重关联被判为"身份断裂"时放弃该配对，任其走认领/新建路径。
# 原行为是把检测仍并进旧 tid → 一个 gid 覆盖两个物理人 → 少算。
_R1_SPLIT_ON_BREAK = os.environ.get("WHT_R1_SPLIT_ON_BREAK", "1") == "1"
# V7 历史重关联的接受成本上限 —— **已采纳，默认 0.40**（设空 = 关闭，回到上游行为）。
#   过门后候选按 cost 升序消费，超过上限的配对直接放弃，让检测落到认领/新建路径。
#   实测 3 例并轨经 V7 以 cost 0.42~0.71 的低证据被接受，本上限正拦这类配对。
#   ⚠ 必须与 R-1 + _CLAIM_THR=0.65 同时开：放开的检测要靠认领路径接回【正确的过期
#     gid】，否则会变成新 id（多算）。单独开实测 ±0，三者同时开实测 +3（34→37）。
#   扫参：0.30 持平 / **0.40 最优** / 0.45 掉到 32。
_HU_V7_COST_MAX = os.environ.get("HU_V7_COST_MAX", "0.40").strip()
# ---- HU 原型实验开关（默认关闭；关闭时行为与原脚本一致）----
# HU_POS_CAP：普通匹配位置门 (u_thresh/v_thresh) 的绝对上限（默认关闭）。
_HU_POS_CAP = os.environ.get("HU_POS_CAP", "").strip()
# HU_HANDOFF_DIST：appearance_proves_handoff 守卫的外观距离阈值（默认 0.48=原版；
#   微调特征尺度下同人 sim 更低，该固定阈值会误伤合法续接，可据此重标定）
_HU_HANDOFF_DIST = float(os.environ.get("HU_HANDOFF_DIST", "0.48"))
# HU_TRACE_SEQ=1：帧级匹配调试记录（写 self._hu_trace_log，仅调试用）
_HU_TRACE = os.environ.get("HU_TRACE_SEQ", "") != ""
# HU_V2_ALWAYS=1：V2 双坐标恢复常开（原版仅窗不稳/贴边时启用）。
_HU_V2_ALWAYS = os.environ.get("HU_V2_ALWAYS", "") == "1"
# HU_WIN_ORDER_FIX：WindowIdTracker 末尾「左右顺序修正」的作用范围。
#   all（默认，= 上游行为）/ fallback（只修「按顺序补配对」支）/ off（关闭）
_HU_WIN_ORDER_FIX = os.environ.get("HU_WIN_ORDER_FIX", "all").strip().lower()
# HU_TRAVEL_MIN_DX：判定行驶方向所需的最小累计车框位移（px）。低于此值退回
#   is_car_front_on_left 判向。
_HU_TRAVEL_MIN_DX = float(os.environ.get("HU_TRAVEL_MIN_DX", "20.0"))

# ---- HU 外观用法改造（默认全关；关闭时行为与原脚本逐行一致）----
# 动机：外观现在只被当成"绝对量与固定常数比较"（_APP_GATE 硬门 + 常数打分），
#   但决策点正负 sim 重叠，任何常数都无解 → 收紧必亏（实测 gate 0.45→0.15
#   计数单调 19→32）。改为在"本帧候选集合内部"做相对判断，随尺度自适应。
# HU_TOPK=k：track 外观档案相似度由 max 改为 top-k 均值（k>=2 生效），默认空=max。
_HU_TOPK = os.environ.get("HU_TOPK", "").strip()
_HU_TOPK = int(_HU_TOPK) if _HU_TOPK else None
# HU_REL_MARGIN=m：启用"相对判据"。每个检测取外观最优轨，若
#   最优-次优 >= m（或该检测只有一个候选 / 最优已过 HU_REL_ABS 确信线），
#   则外观拍板：只保留最优轨、剪掉其余竞争边；否则视为"外观无法区分"→
#   该检测的所有边退回仅按运动打分（外观弃权，位置决定）。
_HU_REL_MARGIN = os.environ.get("HU_REL_MARGIN", "").strip()
_HU_REL_MARGIN = float(_HU_REL_MARGIN) if _HU_REL_MARGIN else None
_HU_REL_ABS = float(os.environ.get("HU_REL_ABS", "0.85"))   # 外观"确信线"
# HU_MNN=1：互选约束（边 (det,track) 保留 ⟺ det 的最优轨是该轨 且 该轨的最优检测是该检测）。
_HU_MNN = os.environ.get("HU_MNN", "") == "1"
_HU_APP_LOGIC = (_HU_REL_MARGIN is not None) or _HU_MNN

# HU_METRIC=ciou：位置度量从"归一化 (u,v) 欧氏距离"换成 CIoU 式度量
#   （中心距离 + 尺度一致性 + 宽高比），针对"相邻帧位移≈目标宽 4.5 倍、纯距离区分度低"。
_HU_METRIC = os.environ.get("HU_METRIC", "").strip().lower()
_CIOU_ALPHA = float(os.environ.get("HU_CIOU_ALPHA", "1.0"))   # 中心距离项权重
_CIOU_BETA = float(os.environ.get("HU_CIOU_BETA", "1.0"))     # 尺度一致性项权重
_CIOU_GAMMA = float(os.environ.get("HU_CIOU_GAMMA", "0.5"))   # 宽高比项权重
# HU_METRIC=shape：位置门放宽倍数 + shape 主门阈值
_HU_SHAPE_POS_RELAX = float(os.environ.get("HU_SHAPE_POS_RELAX", "2.0"))
_HU_SHAPE_GATE = float(os.environ.get("HU_SHAPE_GATE", "0.75"))
# HU_FUSE="w_pos,w_app,w_shape"：三路线索加权融合打分（默认关闭 = 原两路 0.35运动+0.65外观）
#   依据诊断 0.3（oracle 上界实验）：决策点上三条线索各自判别力
#     位置 64.5% / 外观 62.3% / 尺度+长宽比 60.1%，但 **ORACLE 上界 94.2%（差 +29.7 点）**
#     → 三条线索高度互补，融合有可利用空间。本开关把"尺度"作为【独立第三项】加入打分
#     （注意：此前试过的 HU_METRIC=ciou 是【替换】运动项，不是增加第三项，二者不同）。
_HU_MAHAL = float(os.environ.get("HU_MAHAL", "0"))          # 马氏门阈值 K（0=关闭）
_HU_MAHAL_FLOOR = float(os.environ.get("HU_MAHAL_FLOOR", "0.04"))  # σ 下限

# ---------------------------------------------------------------------------
# 方案 A：各向异性自适应门（用户 2026-10-04 提出）
#   上面的 HU_MAHAL 用**单一** σ（对 u、v 一视同仁）。但实测纵向比横向稳 5 倍
#   （同人 |Δv| 中位 0.012 vs |Δu| 中位 0.061），所以 σ_u 与 σ_v 应当分开估。
#   本例形态：轨道 201 自己的 v 只动 0.005，新人 GT3 差 0.074（15 倍）→ 拒绝。
#   注意：σ 由该轨自身历史估计，序列仅 8 帧、轨道常只有 3 次观测，故设下限兜底。
# ---------------------------------------------------------------------------
_HU_MAHAL_ANISO = float(os.environ.get("HU_MAHAL_ANISO", "0"))       # K，0=关闭
_HU_MAHAL_FLOOR_U = float(os.environ.get("HU_MAHAL_FLOOR_U", "0.05"))
_HU_MAHAL_FLOOR_V = float(os.environ.get("HU_MAHAL_FLOOR_V", "0.02"))

# ---------------------------------------------------------------------------
# 方案 B：车窗框时域平滑（用户 2026-10-04 提出）
#   拆解实测：同人 |Δv| 的三项来源里，**车窗框上沿的纵向漂移占 39%**（最大项，
#   折算约 2.4 像素）。而车辆是【横向】行驶，车窗上沿的 y 理论上根本不该变
#   → 这 39% 几乎全是标注/检测的随机抖动，属于"公共项"（同帧同窗的人共享）。
#   与失败的"换参考系"不同：换参考系是全局等比例缩放（信噪比不变），
#   平滑是【减掉一个共享噪声项】（信号不动、噪声下降）。
#   只平滑 y1/y2：x1/x2 随车辆快速平移，平滑会严重滞后；
#   而 v 只依赖 y1 与窗高，宽度平滑对 v 无益。
#   取值：EMA 系数 α，新 = α·旧 + (1−α)·本次。0 = 关闭。
# ---------------------------------------------------------------------------
_HU_WIN_SMOOTH = float(os.environ.get("HU_WIN_SMOOTH", "0"))
_HU_FUSE = os.environ.get("HU_FUSE", "").strip()

# HU_NO_POS_GATE=1：**完全关掉位置门**（阈值放到无穷），但打分里的 motion 项仍按
# 原阈值归一，保持"门"和"分"解耦。用途：做「位置 vs 外观谁主导」的消融对照。
# 与 HU_POS_CAP（收紧）方向相反，是本项目第一次测"彻底拿掉位置门"。
_HU_NO_POS_GATE = os.environ.get("HU_NO_POS_GATE", "0").strip() == "1"

# WHT_REVERSAL_DU：方向反转守卫的位移门槛（上游硬编码 0.10，本副本已采纳 0.08）。
#
# 动机：20260808120053537 的并轨里，202 的人头速度 +0.074（向右），
# 而新人 GT5 的框接上去需要 -0.065（向左）—— 确实是方向反转，
# 但 |Δu|=0.065 < 0.10 没触发守卫，于是新人顶掉了旧轨。
#
# 全 54 集扫参（2026-10-04，配置 HU_V7_COST_MAX=0.40）：
#   0.10(上游) 34/54 | 0.08 35/54（改好1 改坏0） | 0.07 34/54（改好1 改坏1）
#   0.06~0.03  35/54（改好2 改坏1：多修 20260808120053537，
#                        但 20260808140911994 的合法续接在 0.07 就被误伤）
# 0.08 是唯一"零回归"的点，故采纳为默认；0.06 以下需接受一处回归。
_REVERSAL_DU = float(os.environ.get("WHT_REVERSAL_DU", "0.08"))

# HU_V_THRESH：**只收紧纵向门** v_thresh 的绝对上限（默认关闭）。
#
# 动机（用户 2026-10-04 提出，54 集实测支持）：
#   车从画面一侧驶向另一侧 → 横向 u 随行驶匀速漂移（Δu 中位 0.061，p90 0.152）；
#   纵向 v 几乎不动（Δv 中位 0.012，p90 0.034，逐帧斜率只有横向的 7%）。
#   但代码里 v_thresh 是直接照抄 u_thresh 的（W2 都写 0.22）——比纵向最优阈值
#   宽约 7 倍。矩形门(u,v 各自设阈)最优判别力 89.3%，高于欧氏合并 85.0%。
#   反例（20260814081406448 F5）：新人 GT3 顶替轨道 201，Δu=0.001（无缝）
#   而 Δv=0.077（是 201 自身 v 噪声的 15 倍）——纵向本可拦住。
#   注意：旧的 HU_POS_CAP 是 **u、v 同时**压缩，横向门被压坏才导致 25/14/2；
#   本开关只动纵向，横向保持原样。
_HU_V_THRESH = os.environ.get("HU_V_THRESH", "").strip()

# HU_GATE_DEBUG=1：打印每条候选配对在三道门（主匹配 / V2 / V7）上的 du/dv 与阈值，
# 用于定位"某条顶替到底是从哪条路径进来的"。
_HU_GATE_DEBUG = os.environ.get("HU_GATE_DEBUG", "0").strip() == "1"
# HU_TENTATIVE_MISS：hits<2 的弱轨在连续漏检超过该帧数后被踢出匹配候选。
# 上游硬编码 1（= 只观测过 1 次的轨，漏检 2 帧即永久失去接回资格）。调大即放松。
_HU_TENTATIVE_MISS = float(os.environ.get("HU_TENTATIVE_MISS", "1"))
# HU_TENTATIVE_APP：把上面的守卫条件化 —— 弱轨被踢出前先看外观，相似度 >= 该阈值
# 才允许复活。0 = 关闭（保持上游行为）。
_HU_TENTATIVE_APP = float(os.environ.get("HU_TENTATIVE_APP", "0"))
# HU_LOWCONF_CREATE_MIN：低分框**不得建新轨**的置信度底线（0 = 关闭，保持上游行为）。
# 语义 = ByteTrack 的安全属性：低于该值的检测只能**延长**已有轨道，匹配不上即丢弃。
# 配套用法：低阈值检测缓存（CACHE_CONF_RAW=0.10）+ 本开关 0.35 —— 此时 ≥0.35 的框
# 建轨行为与基线一致，多出来的低分框只参与"延长"，不产生任何可被计数的新身份。
_HU_LOWCONF_CREATE_MIN = float(os.environ.get("HU_LOWCONF_CREATE_MIN", "0"))
# HU_TWOPASS：真·两阶段关联（ByteTrack 形式，0 = 关闭）。
# 第 1 趟只用 conf >= 该阈值的检测做全局指派 —— 与基线逐位相同；
# 第 2 趟用低于该阈值的检测，只对**第 1 趟没配上的**轨道做指派。
# 与 HU_LOWCONF_CREATE_MIN 的差别：后者仍让低分框与高分框**同场竞争**，
# 会抢走本该归高分框的轨道；本开关把两趟严格分开，高分框优先。
_HU_TWOPASS = float(os.environ.get("HU_TWOPASS", "0"))
_HU_V_CAP = float(_HU_V_THRESH) if _HU_V_THRESH else None

# 纵向门的三个落点（实测 20260814081406448 F5：主门拒了，V7 又用 dv<=0.32 接回来）
_V2_MAX_DV = 0.36 if _HU_V_CAP is None else min(0.36, _HU_V_CAP)
_V7_MAX_DV = 0.32 if _HU_V_CAP is None else min(0.32, _HU_V_CAP)
if _HU_FUSE:
    _fw = [float(x) for x in _HU_FUSE.split(",")]
    _FUSE_W = (_fw + [0.0, 0.0, 0.0])[:3] if len(_fw) >= 3 else (0.35, 0.65, 0.0)
else:
    _FUSE_W = None

# ---- HU 计数级占用约束（建议6，默认关闭）----
# 动机：逐帧配对已被证伪（宽松匹配最优）。剩余的"同座位交接并轨"是计数层问题：
#   一个 gid 的观测轨迹在时间上出现"跳变"（相邻观测位移远超该轨自身的典型速度），
#   且跳变发生在一段没有共现观测的空档之后 → 该 gid 覆盖了不止一个物理人。
#   宽松匹配仍有价值（它兜住"位置已过门、外观分不清"的合法续接），本块不改匹配，
#   只在计数时把这类"容纳了两个人"的 gid 按段拆开，改回裸 distinct。
# HU_OCC=1 启用；HU_OCC_COEXIST：判定"与另一 gid 同帧近距冲突"的 (u,v) 距离上限
#   （0 = 只认"同 gid 同帧多观测"这一最强证据；>0 时额外把两轨同帧近距视为共现冲突）。
_HU_OCC = os.environ.get("HU_OCC", "") == "1"
_HU_OCC_COEXIST = float(os.environ.get("HU_OCC_COEXIST", "0.0"))
_HU_OCC_GAP = int(os.environ.get("HU_OCC_GAP", "1"))
_HU_OCC_K = float(os.environ.get("HU_OCC_K", "3.0"))

_SIDE_STAT = {"gate_reject": 0, "claim": 0, "handoff_skip": 0,
              "rel_prune": 0, "mnn_prune": 0, "occ_split": 0,
              "pos_pass": 0, "app_reject": 0, "app_checked": 0,
              "lowconf_drop": 0}
_SIDE_SIMS = []          # 每个"过了位置门、且外观可比"的配对的 sim（供门限诊断）


def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


class WindowHeadTracker:
    def __init__(self, window_id=1, iou_thresh=0.05):
        self.window_id = window_id
        self.next_head_id = 1
        self.tracked_heads = {}
        self.track_summaries = {}
        self.unique_head_ids = set()
        self.frame_index = 0
        self.frame_detected_counts = []
        # Per-frame vehicle/global motion passed by WindowedHeadCounter.  It is
        # retained even on missed-head frames so fragmented tracks can be
        # compared against a multi-frame motion prediction instead of a fixed
        # frame-gap rule.
        self.frame_motion_shifts = {}
        # Per-frame observability is independent of total sequence length.
        # It lets a long but fast sequence use sparse-observation safeguards.
        self.frame_window_boxes = {}
        self.frame_observation_quality = {}
        self.max_ttl = 6
        self.prev_win_box = None

        # V19: 强历史重关联的“同ID内部身份断裂”证据。
        # key=(track_id, reappear_frame)，避免 get_count() 多次调用重复累计。
        self.history_reassoc_identity_breaks = set()

        # V67: 保存所有实际发生的历史重关联细节，供最终计数层判断
        # LOST 后同ID是否已经从稳定锚点一侧切换到另一侧。
        self.history_reassoc_records = []
        self.verified_handoff_ids = set()

        # ============================================================
        # V40: 全局共存硬约束 / coexistence graph
        #
        # 两个 Track ID 只要曾经在同一帧被实际观测到，就永久证明：
        #     ID_A != ID_B
        #
        # 后续任何 stitch / peak reconciliation / 物理身份压缩都禁止
        # 把这两个 ID 合并。使用 frozenset 与方向无关。
        # ============================================================
        self.cannot_merge_pairs = set()

        # ---- sideview 式双门控：外观档案（每个 gid 最近 _APP_HIST 帧外观）----
        # 门限打分与"TTL 过期认领"都查这份档案；track 删除后档案仍保留该 gid。
        self.app_archive = {}

        # HU 计数级占用约束：{tid: [(frame,u,v), ...]} 该 gid 的逐帧观测（含共现帧）
        self._occ_clean_obs = {}

    def _ciou_motion(self, du, dv, dist_uv, _diag, rw_c, rw_p, rh_c, rh_p):
        """CIoU 式几何度量（归一化窗坐标空间）→ 返回 [0,1] 的相似分（越大越像同一人）。

        与本脚本原 "1 - dist/_diag" 的差别：额外引入
          · 尺度一致性：两次观测的框面积比（同人相邻帧尺度应接近；车在靠近时单调变）
          · 宽高比一致性：人头长宽比在帧间应稳定
        这三项都是纯几何、不依赖外观，且对"相邻帧位移≈目标宽 4.5 倍、纯中心距离区分度低"
        的情形提供额外判别维度（人头的尺度/长宽比在帧间比中心位置稳定得多）。
        """
        center = 1.0 - min(1.0, dist_uv / _diag)                      # 中心距离项
        a_c = max(1e-9, rw_c * rh_c)
        a_p = max(1e-9, rw_p * rh_p)
        scale = min(a_c, a_p) / max(a_c, a_p)                          # 面积比 ∈ (0,1]
        ar_c = rw_c / max(1e-9, rh_c)
        ar_p = rw_p / max(1e-9, rh_p)
        ar = min(ar_c, ar_p) / max(max(ar_c, ar_p), 1e-9)              # 长宽比之比 ∈ (0,1]
        num = _CIOU_ALPHA * center + _CIOU_BETA * scale + _CIOU_GAMMA * ar
        den = _CIOU_ALPHA + _CIOU_BETA + _CIOU_GAMMA
        return float(num / den)

    def _archive_appearance(self, gid, feat):
        """记录一个 gid 的一次外观观测，档案保留最近 _APP_HIST 帧。"""
        if feat is None:
            return
        q = self.app_archive.get(int(gid))
        if q is None:
            q = []
            self.app_archive[int(gid)] = q
        q.append(np.asarray(feat, dtype=np.float32))
        if len(q) > _APP_HIST:
            del q[:len(q) - _APP_HIST]

    def _max_sim(self, gid, feat):
        """新检测外观与某 gid 最近 _APP_HIST 帧外观的相似度（无档案/无特征返回 None）。
        HU_TOPK 设定时取 top-k 均值（抑制 max 对冒充者的系统性抬高），否则取 max（原行为）。"""
        if feat is None:
            return None
        q = self.app_archive.get(int(gid))
        if not q:
            return None
        vals = [float(np.clip(float(np.dot(feat, f)), -1.0, 1.0)) for f in q]
        if _HU_TOPK and len(vals) > 1:
            vals = sorted(vals, reverse=True)[:_HU_TOPK]
            return float(sum(vals) / len(vals))
        return max(vals)

    def _claim_expired_identity(self, c_head, c_idx, curr_heads,
                                matched_curr, matched_prev_ids):
        """sideview 阶段二：外观门匹配漏掉的检测，若与某个"已过期(无存活轨)的 gid"高度相似，
        则认领回该旧 gid（重建同 gid 轨迹），避免 TTL 过后同一个人被开成新 id 造成虚增。
        约束：被认领 gid 不能与"本帧已在场的任一 id"存在历史共存冲突。"""
        feat = c_head.get('appearance')
        if feat is None or not _ENABLE_CLAIM or _MATCH_MODE != "sidegate":
            return None
        present_ids = set(matched_prev_ids)
        best = None
        for gid in list(self.app_archive):
            gid = int(gid)
            if gid in self.tracked_heads:
                continue                       # 还有存活轨 → 交给位置/历史重关联
            if gid in present_ids or gid in matched_prev_ids:
                continue
            # 若该过期 gid 曾与"本帧在场 id"同框（物理上不同人），不能再认领 → 否则同一人分身
            conflict = False
            for pid in present_ids:
                if frozenset((gid, int(pid))) in self.cannot_merge_pairs:
                    conflict = True
                    break
            if conflict:
                continue
            sim = self._max_sim(gid, feat)
            if sim is None or sim < _CLAIM_THR:
                continue
            if best is None or sim > best[0]:
                best = (sim, gid)
        if best is None:
            return None
        _, gid = best
        return gid

    def _mark_coexisting_ids(self, frame):
        """Persist all ID pairs that were actually observed in the same frame."""
        observed_ids = []
        for tid, info in self.track_summaries.items():
            observations = info.get('observations', {})
            if int(frame) in observations:
                observed_ids.append(int(tid))

        observed_ids = sorted(set(observed_ids))
        for i in range(len(observed_ids)):
            for j in range(i + 1, len(observed_ids)):
                pair = frozenset((observed_ids[i], observed_ids[j]))
                if pair not in self.cannot_merge_pairs:
                    self.cannot_merge_pairs.add(pair)
                    print(
                        f"[共存硬约束] W{self.window_id} | "
                        f"frame={frame} | "
                        f"ID{observed_ids[i]} != ID{observed_ids[j]}"
                    )

    def _cannot_merge(self, id_a, id_b):
        """
        V43 FIX: 共存约束扩展为传递冲突保护。

        场景：
            frame中同时出现 ID201(ID锚点)、ID202、ID203
            已记录：
                ID201 != ID202
                ID201 != ID203

        后续N2碎轨保护错误判断：
                ID202 -> ID203
        会把两个真实乘客合并。

        如果两个ID都与同一个历史ID存在共存关系，则它们也不能合并。
        这样保持 V40 coexistence graph 的物理身份约束。
        """
        if id_a == id_b:
            return False

        a = int(id_a)
        b = int(id_b)

        if frozenset((a, b)) in self.cannot_merge_pairs:
            return True

        # transitive coexistence protection
        neighbors_a = set()
        neighbors_b = set()
        for pair in self.cannot_merge_pairs:
            if a in pair:
                neighbors_a.update(pair)
            if b in pair:
                neighbors_b.update(pair)

        neighbors_a.discard(a)
        neighbors_b.discard(b)

        if neighbors_a.intersection(neighbors_b):
            return True

        return False

    def update(self, heads_in_window, window_box=None, motion_shift=(0.0, 0.0), is_head_on_left=True, img_w=1920, appearance_features=None, head_confs=None, person_probs=None):
        self.frame_index += 1
        try:
            self.frame_motion_shifts[self.frame_index] = (
                float(motion_shift[0]), float(motion_shift[1]))
        except (TypeError, ValueError, IndexError):
            self.frame_motion_shifts[self.frame_index] = (0.0, 0.0)
        if window_box is not None:
            box_now = tuple(float(v) for v in window_box)
            win_w_now = max(1.0, box_now[2] - box_now[0])
            win_h_now = max(1.0, box_now[3] - box_now[1])
            prev_box = self.frame_window_boxes.get(self.frame_index - 1)
            width_change = height_change = centre_residual = 0.0
            if prev_box is not None:
                prev_w = max(1.0, prev_box[2] - prev_box[0])
                prev_h = max(1.0, prev_box[3] - prev_box[1])
                width_change = abs(win_w_now - prev_w) / prev_w
                height_change = abs(win_h_now - prev_h) / prev_h
                shift_x, shift_y = self.frame_motion_shifts[self.frame_index]
                predicted_x = (prev_box[0] + prev_box[2]) * 0.5 + shift_x
                predicted_y = (prev_box[1] + prev_box[3]) * 0.5 + shift_y
                centre_residual = float(np.hypot(
                    (box_now[0] + box_now[2]) * 0.5 - predicted_x,
                    (box_now[1] + box_now[3]) * 0.5 - predicted_y
                )) / max(win_w_now, win_h_now)
            shift_x, shift_y = self.frame_motion_shifts[self.frame_index]
            motion_ratio = float(np.hypot(shift_x, shift_y)) / win_w_now
            clipped = box_now[0] <= 8.0 or box_now[2] >= float(img_w) - 8.0
            unstable = (width_change > 0.15 or height_change > 0.20 or
                        centre_residual > 0.18 or clipped)
            self.frame_window_boxes[self.frame_index] = box_now
            self.frame_observation_quality[self.frame_index] = {
                "motion_ratio": motion_ratio,
                "width_change": width_change,
                "height_change": height_change,
                "centre_residual": centre_residual,
                "clipped": bool(clipped),
                "unstable": bool(unstable),
            }
        
        for tid in list(self.tracked_heads.keys()):
            self.tracked_heads[tid]['ttl'] -= 1
            self.tracked_heads[tid]['missed_frames'] = self.tracked_heads[tid].get('missed_frames', 0) + 1
            if self.tracked_heads[tid]['ttl'] <= 0:
                del self.tracked_heads[tid]

        if window_box is None or not heads_in_window:
            self.frame_detected_counts.append(0)
            if window_box is not None:
                self.prev_win_box = window_box
            return self.count_distinct()

        curr_frame_count = len(heads_in_window)
        self.frame_detected_counts.append(curr_frame_count)

        win_x1, win_y1, win_x2, win_y2 = window_box
        win_w = max(1.0, win_x2 - win_x1)
        win_h = max(1.0, win_y2 - win_y1)

        is_window_unstable = False
        if self.prev_win_box is not None:
            prev_w = max(1.0, self.prev_win_box[2] - self.prev_win_box[0])
            w_diff_ratio = abs(win_w - prev_w) / prev_w
            if w_diff_ratio > 0.15:
                is_window_unstable = True

        # FIX-20260827:
        # When a physical window reaches the image edge, the detected window box
        # is clipped.  The normalized head coordinate
        #     u = (head_cx - win_x1) / win_w
        # then changes coordinate system abruptly even for the same passenger.
        # Mark this state explicitly so identity recovery can avoid treating the
        # artificial u jump as a new person.
        edge_margin = 8.0
        is_window_clipped = (
            win_x1 <= edge_margin or
            win_x2 >= img_w - edge_margin
        )

        self.prev_win_box = window_box

        curr_heads = []
        if appearance_features is None:
            appearance_features = [None] * len(heads_in_window)
        if head_confs is None:
            head_confs = [None] * len(heads_in_window)
        if person_probs is None:
            person_probs = [None] * len(heads_in_window)
        for head_index, h in enumerate(heads_in_window):
            w = max(1.0, h[2] - h[0])
            h_len = max(1.0, h[3] - h[1])
            cx = (h[0] + h[2]) / 2.0
            cy = (h[1] + h[3]) / 2.0
            
            u = (cx - win_x1) / win_w
            v = (cy - win_y1) / win_h

            curr_heads.append({
                'box': h, 'cx': cx, 'cy': cy, 'w': w, 'h': h_len,
                'u': u, 'v': v,
                'rw': w / win_w, 'rh': h_len / win_h,
                'appearance': appearance_features[head_index],
                'conf': head_confs[head_index],
                'person_prob': person_probs[head_index]
            })

        def update_track(tid, c_head):
            """Update one track and confirm it only after two observations."""
            p_info = self.tracked_heads[tid]
            elapsed = max(1, p_info.get('missed_frames', 1))
            observed_du = (c_head['u'] - p_info['u']) / elapsed
            observed_dv = (c_head['v'] - p_info['v']) / elapsed
            observed_dcx = (c_head['cx'] - p_info['cx']) / elapsed
            observed_dcy = (c_head['cy'] - p_info['cy']) / elapsed
            old_du = p_info.get('du_velocity', 0.0)
            old_dv = p_info.get('dv_velocity', 0.0)
            old_dcx = p_info.get('dcx_velocity', 0.0)
            old_dcy = p_info.get('dcy_velocity', 0.0)
            # HU_MAHAL：本次匹配的预测残差（用【更新前】的位置+速度预测）
            _pred_resid = max(
                abs(c_head['u'] - (p_info['u'] + old_du * elapsed)),
                abs(c_head['v'] - (p_info['v'] + old_dv * elapsed)))
            # 方案 A：横纵分开的预测残差（供各向异性马氏门使用）
            _resid_u = abs(c_head['u'] - (p_info['u'] + old_du * elapsed))
            _resid_v = abs(c_head['v'] - (p_info['v'] + old_dv * elapsed))
            hits = p_info.get('hits', 1) + 1
            old_appearance = p_info.get('appearance')
            new_appearance = c_head.get('appearance')
            appearance_distance = None
            if old_appearance is not None and new_appearance is not None:
                appearance_distance = float(1.0 - np.clip(
                    np.dot(old_appearance, new_appearance), -1.0, 1.0))

            p_info.update({
                'u': c_head['u'], 'v': c_head['v'],
                'cx': c_head['cx'], 'cy': c_head['cy'],
                'w': c_head['w'], 'h': c_head['h'],
                'rw': c_head['rw'], 'rh': c_head['rh'],
                # HU_MAHAL：维护该轨自身的"预测误差"统计（马氏门用）
                # 残差 = 匹配时 |观测 - 预测|（在更新位置【之前】算）
                'resid_ema': (
                    None if _HU_MAHAL <= 0 else float(
                        0.7 * p_info.get('resid_ema', _pred_resid) + 0.3 * _pred_resid)
                ),
                # 方案 A：横纵各自的自适应尺度（σ_u / σ_v）
                'resid_ema_u': (
                    None if _HU_MAHAL_ANISO <= 0 else float(
                        0.7 * p_info.get('resid_ema_u', _resid_u) + 0.3 * _resid_u)
                ),
                'resid_ema_v': (
                    None if _HU_MAHAL_ANISO <= 0 else float(
                        0.7 * p_info.get('resid_ema_v', _resid_v) + 0.3 * _resid_v)
                ),
                'du_velocity': float(np.clip(
                    0.60 * old_du + 0.40 * observed_du, -0.20, 0.20
                )),
                'dv_velocity': float(np.clip(
                    0.60 * old_dv + 0.40 * observed_dv, -0.15, 0.15
                )),
                # Absolute image-coordinate velocity. This remains meaningful
                # when the detected window box changes width or is clipped.
                'dcx_velocity': float(np.clip(
                    0.35 * old_dcx + 0.65 * observed_dcx, -400.0, 400.0
                )),
                'dcy_velocity': float(np.clip(
                    0.35 * old_dcy + 0.65 * observed_dcy, -250.0, 250.0
                )),
                'hits': hits,
                'ttl': self.max_ttl,
                'missed_frames': 0
            })
            if new_appearance is not None:
                p_info['appearance'] = new_appearance
                self._archive_appearance(tid, new_appearance)
            if hits >= 2:
                self.unique_head_ids.add(tid)
            summary = self.track_summaries[tid]
            summary.update({
                'last_frame': self.frame_index,
                'last_u': c_head['u'], 'last_v': c_head['v'],
                'last_rw': c_head['rw'], 'last_rh': c_head['rh'],
                'du_velocity': p_info['du_velocity'],
                'dv_velocity': p_info['dv_velocity'],
                'last_cx': c_head['cx'], 'last_cy': c_head['cy'],
                'dcx_velocity': p_info['dcx_velocity'],
                'dcy_velocity': p_info['dcy_velocity']
            })
            summary['frames'].add(self.frame_index)
            summary['observations'][self.frame_index] = {
                'u': c_head['u'], 'v': c_head['v'],
                'rw': c_head['rw'], 'rh': c_head['rh'],
                'cx': c_head['cx'], 'cy': c_head['cy'],
                'w': c_head['w'], 'h': c_head['h'],
                'conf': c_head.get('conf'),
                'person_prob': c_head.get('person_prob')
            }
            if appearance_distance is not None:
                summary.setdefault('appearance_distances', []).append(
                    (self.frame_index, appearance_distance))
            if new_appearance is not None:
                summary['last_appearance'] = new_appearance
                summary.setdefault('appearances', {})[self.frame_index] = new_appearance

        curr_heads.sort(key=lambda item: item['u'])
        track_ids = list(self.tracked_heads.keys())
        track_ids.sort(key=lambda tid: self.tracked_heads[tid]['u'])

        num_curr = len(curr_heads)
        num_prev = len(track_ids)

        matched_curr = set()
        matched_prev_ids = set()
        verified_handoff_curr = {}
        handoff_blocked_tracks = set()

        # ---- HU debug trace（env HU_TRACE_SEQ 时记录帧级匹配细节）----
        _hu_dets = [(i, round(h['u'], 3), round(h['v'], 3),
                     round(h.get('rw', 0.0), 4), round(h.get('rh', 0.0), 4))
                    for i, h in enumerate(curr_heads)]
        _hu_tracks_pre = [{'tid': int(t), 'u': round(self.tracked_heads[t]['u'], 3),
                           'v': round(self.tracked_heads[t]['v'], 3),
                           'rw': round(self.tracked_heads[t].get('rw', 0.0), 4),
                           'rh': round(self.tracked_heads[t].get('rh', 0.0), 4),
                           'duv': round(self.tracked_heads[t].get('du_velocity', 0.0), 3),
                           'hits': self.tracked_heads[t].get('hits', 1),
                           'miss': self.tracked_heads[t].get('missed_frames', 0)}
                          for t in track_ids]
        _hu_pairs, _hu_matched, _hu_newids = [], [], []
        _occ_obs = {}      # HU_OCC: {tid: [(frame,u,v), ...]} 干净单占帧观测

        if num_prev > 0:
            candidates = []
            _pair_sim, _pair_mot = {}, {}   # (c_idx,tid) -> 外观sim / motion分（HU 外观改造用）

            for c_idx, c_head in enumerate(curr_heads):
                for tid in track_ids:
                    p_info = self.tracked_heads[tid]

                    # A one-frame tentative detection is not a reliable
                    # identity anchor.  If it disappears for a full frame,
                    # do not let a different passenger at a nearby position
                    # resurrect and confirm that old track.
                    # HU_TENTATIVE_MISS：(HU 实验开关) 上面那条守卫的漏检帧数上限，
                    # 上游硬编码为 1。调大 = 允许"只观测过 1 次"的轨道在更长空档后
                    # 仍可被接回（代价：也可能被另一个乘客顶替）。
                    # HU_TENTATIVE_APP：(HU 实验开关) 把上面那条守卫**条件化** ——
                    # 弱轨被踢出前先比对"新检测外观 vs 该弱轨仅有的那一次观测外观"，
                    # 相似度 >= 阈值才允许复活。目的：把"同一个人漏检后回来"与
                    # "另一个人顶替进来"分开，打破 HU_TENTATIVE_MISS 的对称抵消。
                    # 0 = 关闭（完全保持上游行为）。
                    if (p_info.get('hits', 1) < 2 and
                            p_info.get('missed_frames', 0) > _HU_TENTATIVE_MISS):
                        if _HU_TENTATIVE_APP <= 0:
                            continue
                        _a_new = c_head.get('appearance')
                        _a_old = p_info.get('appearance')
                        if _a_new is None or _a_old is None:
                            continue
                        if float(np.dot(_a_new, _a_old)) < _HU_TENTATIVE_APP:
                            continue

                    elapsed = max(1, p_info.get('missed_frames', 1))
                    pred_u = p_info['u'] + p_info.get('du_velocity', 0.0) * elapsed
                    pred_v = p_info['v'] + p_info.get('dv_velocity', 0.0) * elapsed
                    du = abs(c_head['u'] - pred_u)
                    dv = abs(c_head['v'] - pred_v)

                    observed_du = (c_head['u'] - p_info['u']) / elapsed
                    previous_du = p_info.get('du_velocity', 0.0)
                    raw_position_jump = (
                        (c_head['u'] - p_info['u']) ** 2 +
                        (c_head['v'] - p_info['v']) ** 2
                    ) ** 0.5
                    # Identity evidence is evaluated in a strict order:
                    # first require a residual X/Y movement after trajectory
                    # prediction; only then inspect appearance.  This avoids
                    # splitting a stationary passenger because lighting or
                    # confidence changed its lightweight feature.
                    position_changed = (
                        raw_position_jump >= 0.05 and
                        max(du, dv) >= 0.045
                    )
                    appearance_distance = None
                    appearance_similarity = None
                    # HU 计数级占用约束：此处不做采集（一对 (det,track) 会被逐帧重复
                    # 评估，逐对采样会重复计数）。改为在帧末 matched_curr 定稿后，
                    # 按"每个 gid 每帧一次"采集真实观测。见本帧匹配结束处。

                    if (c_head.get('appearance') is not None and
                            p_info.get('appearance') is not None):
                        appearance_similarity = float(np.clip(
                            np.dot(c_head['appearance'], p_info['appearance']),
                            -1.0, 1.0))
                        if position_changed:
                            appearance_distance = 1.0 - appearance_similarity
                    # A newcomer can occupy almost exactly the previous
                    # passenger's predicted seat. In that case geometry alone
                    # is misleading: the closer the raw pane-relative slot,
                    # the stronger the appearance agreement required. Apply
                    # this only to an already-confirmed track and require an
                    # independent scale discontinuity (or extremely low
                    # appearance), so ordinary viewpoint changes on tentative
                    # tracks are unaffected.
                    prev_area = max(
                        1.0,
                        float(p_info.get('w', c_head['w'])) *
                        float(p_info.get('h', c_head['h'])))
                    curr_area = max(1.0, float(c_head['w']) * float(c_head['h']))
                    area_ratio = max(prev_area, curr_area) / min(prev_area, curr_area)
                    same_slot_required_sim = max(
                        0.72, 0.78 - 0.60 * raw_position_jump)
                    # When a moving vehicle carries a head behind a window
                    # pillar/edge, a full-head box often becomes a narrow
                    # half-head box.  OSNet similarity then drops and the area
                    # ratio grows, but height, vehicle-compensated motion and
                    # pane slot remain continuous.  Treat that asymmetric
                    # width-only shrink as occlusion, not a new occupant.
                    prev_w = max(1.0, float(p_info.get('w', c_head['w'])))
                    prev_h = max(1.0, float(p_info.get('h', c_head['h'])))
                    curr_w = max(1.0, float(c_head['w']))
                    curr_h = max(1.0, float(c_head['h']))
                    shift_x, shift_y = self.frame_motion_shifts.get(
                        self.frame_index, (0.0, 0.0))
                    vehicle_residual = float(np.hypot(
                        float(c_head['cx']) - (float(p_info['cx']) + float(shift_x)),
                        float(c_head['cy']) - (float(p_info['cy']) + float(shift_y)))) / max(
                            prev_h, curr_h, 1.0)
                    velocity_residual = float(np.hypot(
                        float(c_head['cx']) - (float(p_info['cx']) +
                                               float(p_info.get('dcx_velocity', shift_x))),
                        float(c_head['cy']) - (float(p_info['cy']) +
                                               float(p_info.get('dcy_velocity', shift_y))))) / max(
                            prev_h, curr_h, 1.0)
                    previous_observation = self.track_summaries.get(
                        int(tid), {}).get('observations', {}).get(
                            self.frame_index - elapsed, {})
                    previous_conf = float(previous_observation.get('conf') or 0.0)
                    partial_edge_continuation = (
                        raw_position_jump <= 0.08 and
                        curr_w / prev_w <= 0.75 and
                        0.82 <= curr_h / prev_h <= 1.18 and
                        min(vehicle_residual, velocity_residual) <= 0.80 and
                        previous_conf >= 0.50 and
                        float(c_head.get('conf') or 0.0) <= 0.60 * previous_conf)
                    frame_quality = self.frame_observation_quality.get(
                        self.frame_index, {})
                    # Vehicle approach, abrupt window-box changes and image-edge
                    # clipping can change both crop scale and OSNet appearance for
                    # the same passenger. In those frames, weak appearance alone
                    # is not enough to declare a same-seat handoff.
                    high_motion_or_window_change = (
                        bool(frame_quality.get("unstable", False)) or
                        float(frame_quality.get("motion_ratio", 0.0)) >= 0.18)
                    same_slot_appearance_handoff = (
                        p_info.get('hits', 1) >= 2 and
                        elapsed == 1 and
                        max(self.frame_detected_counts, default=0) <= 2 and
                        raw_position_jump <= 0.10 and
                        appearance_similarity is not None and
                        appearance_similarity < same_slot_required_sim and
                        (area_ratio >= 1.25 or appearance_similarity < 0.62) and
                        not partial_edge_continuation and
                        not high_motion_or_window_change)
                    if (partial_edge_continuation and
                            appearance_similarity is not None and
                            appearance_similarity < same_slot_required_sim and
                            GENERAL_COUNT_TRACE):
                        print(f"[partial-head continuation] W{self.window_id} "
                              f"frame={self.frame_index} ID{tid} "
                              f"slot={raw_position_jump:.3f} "
                              f"app={appearance_similarity:.3f}/"
                              f"{same_slot_required_sim:.3f} "
                              f"width={curr_w / prev_w:.3f} "
                              f"height={curr_h / prev_h:.3f} "
                              f"motion={min(vehicle_residual, velocity_residual):.3f}")
                    if same_slot_appearance_handoff:
                        verified_handoff_curr.setdefault(c_idx, set()).add(int(tid))
                        handoff_blocked_tracks.add(int(tid))
                        print(
                            f"[同槽位外观换人拆分] W{self.window_id} "
                            f"frame={self.frame_index} ID{tid} -> new "
                            f"slot={raw_position_jump:.3f} "
                            f"app={appearance_similarity:.3f}/"
                            f"{same_slot_required_sim:.3f} "
                            f"area={area_ratio:.2f}"
                        )
                        continue
                    # A crowded dense sequence can replace one passenger with
                    # another at a similar seat position across a single missed
                    # frame. Split only when motion and appearance independently
                    # show a discontinuity; ordinary viewpoint changes therefore
                    # continue to use the existing ID.
                    # Evaluate the whole candidate set before declaring a
                    # gap handoff.  In crowded frames a wrong candidate may
                    # look discontinuous while another detection is the old
                    # occupant's clear reappearance.  A high-similarity,
                    # same-slot alternative vetoes the split; this prevents
                    # iteration order from creating a hidden extra identity.
                    strong_short_gap_continuity = False
                    if (p_info.get('missed_frames', 0) == 2 and
                            p_info.get('appearance') is not None):
                        for alt_head in curr_heads:
                            alt_feat = alt_head.get('appearance')
                            if alt_feat is None:
                                continue
                            alt_sim = float(np.clip(np.dot(
                                alt_feat, p_info['appearance']), -1.0, 1.0))
                            alt_slot = float(np.hypot(
                                float(alt_head.get('u', 0.0)) -
                                float(p_info.get('u', 0.0)),
                                float(alt_head.get('v', 0.0)) -
                                float(p_info.get('v', 0.0))))
                            if alt_sim >= 0.86 and alt_slot <= 0.10:
                                strong_short_gap_continuity = True
                                break
                    stable_gap_handoff = (
                        p_info.get('hits', 1) >= 2 and
                        p_info.get('missed_frames', 0) == 2 and
                        curr_frame_count >= 3 and
                        position_changed and
                        max(du, dv) >= 0.12 and
                        appearance_distance is not None and
                        appearance_distance >= 0.22 and
                        not strong_short_gap_continuity)
                    if stable_gap_handoff:
                        verified_handoff_curr.setdefault(c_idx, set()).add(int(tid))
                        handoff_blocked_tracks.add(int(tid))
                        print(
                            f"[稳定轨迹换人拆分] W{self.window_id} "
                            f"frame={self.frame_index} ID{tid} -> new "
                            f"resid={max(du, dv):.3f} "
                            f"app_dist={appearance_distance:.3f}"
                        )
                        continue
                    # A pane that has already shown three simultaneous heads
                    # has strong coexistence evidence. Ordinary appearance
                    # handoffs must not inflate it to a fourth identity.
                    low_occupancy_handoff = (
                        max(self.frame_detected_counts, default=0) <= 2
                    )
                    appearance_proves_handoff = (
                        self.window_id == 1 and
                        p_info.get('hits', 1) < 2 and
                        low_occupancy_handoff and
                        position_changed and
                        appearance_distance is not None and
                        appearance_distance >= _HU_HANDOFF_DIST
                    )
                    if appearance_proves_handoff:
                        _SIDE_STAT["handoff_skip"] += 1
                        continue
                    sharp_direction_reversal = (
                        p_info.get('hits', 1) >= 2 and
                        elapsed == 1 and
                        abs(previous_du) >= 0.02 and
                        abs(observed_du) >= _REVERSAL_DU and
                        previous_du * observed_du < 0
                    )
                    if sharp_direction_reversal and not (is_window_unstable or is_window_clipped):
                        continue

                    if self.window_id == 1:
                        u_thresh = 0.40 if is_window_unstable else 0.35
                        v_thresh = 0.30
                    else:
                        u_thresh = 0.35 if is_window_unstable else 0.22
                        v_thresh = 0.28 if is_window_unstable else 0.22

                    # HU 实验：只压纵向门（横向不动）。放在"漏帧放宽"之前，
                    # 让丢失 1-2 帧的轨道仍能拿到额外的纵向余量。
                    if _HU_V_THRESH:
                        v_thresh = min(v_thresh, float(_HU_V_THRESH))

                    # Permit a little more uncertainty after a missed frame,
                    # while keeping the gate finite so a different passenger
                    # is not forcibly attached to the nearest old track.
                    u_thresh += 0.03 * min(max(0, elapsed - 1), 3)
                    v_thresh += 0.02 * min(max(0, elapsed - 1), 3)

                    # HU 原型实验开关：位置门绝对上限（默认关闭）。
                    # 实测剩余并轨多为"同座位交接"（Δu 0.07~0.26），而同人续接 Δu≈0，
                    # 用绝对上限压缩松门，找"收掉交接但不动续接"的取值。
                    if _HU_POS_CAP:
                        _cap = float(_HU_POS_CAP)
                        u_thresh = min(u_thresh, _cap)
                        v_thresh = min(v_thresh, _cap)

                    # HU_METRIC=shape_add：**位置门保持原样**（不加宽），shape 作为【附加】判据串联。
                    #   门 = 原位置门 AND shape 分 >= _HU_SHAPE_GATE
                    #   打分 = 原 sidegate 的打分（motion + appearance），不变
                    # 与 shape / shape_nopos 的区别：那两个都改动了位置门本身。
                    if _HU_METRIC == "shape":
                        u_thresh *= _HU_SHAPE_POS_RELAX
                        v_thresh *= _HU_SHAPE_POS_RELAX
                    if _HU_METRIC == "shape_nopos":
                        u_thresh = 1e9
                        v_thresh = 1e9

                    # HU_MAHAL：马氏距离门 —— 门宽随该轨自身的预测不确定性自适应。
                    # 依据（文献双证：#2 熊豪 式3-2/3-9/3-10、#3 公贺）：
                    # 固定门在"轨迹不确定时"过严（切掉真续接）、"轨迹确定时"过松（放进冒充者）；
                    # 马氏门用轨迹自身的残差 σ 归一化 → 自适应。
                    # 注意：恒速【预测】本来就已有（pred_u/pred_v），本开关只改【门】的形式。
                    # 打分归一化尺度：始终按「原阈值」算，不受下面门开关影响，
                    # 这样"关掉门"只改准入、不改分数量纲（门与分解耦）。
                    _diag_ref = max(1e-6, (u_thresh ** 2 + v_thresh ** 2) ** 0.5)

                    _gate_pass = (du <= u_thresh and dv <= v_thresh)
                    if _HU_NO_POS_GATE:
                        _gate_pass = True          # 消融：位置完全不过滤
                    if _HU_MAHAL > 0:
                        _re = p_info.get('resid_ema')
                        if _re is not None:
                            _sig = max(_HU_MAHAL_FLOOR, float(_re))
                            _mahal = ((du / _sig) ** 2 + (dv / _sig) ** 2) ** 0.5
                            _gate_pass = (_mahal <= _HU_MAHAL)
                    # 方案 A：各向异性马氏门 —— σ_u 与 σ_v 分开，用该轨自身的残差尺度
                    if _HU_MAHAL_ANISO > 0:
                        _su = p_info.get('resid_ema_u')
                        _sv = p_info.get('resid_ema_v')
                        if _su is not None and _sv is not None:
                            _sig_u = max(_HU_MAHAL_FLOOR_U, float(_su))
                            _sig_v = max(_HU_MAHAL_FLOOR_V, float(_sv))
                            _mahal = ((du / _sig_u) ** 2 +
                                      (dv / _sig_v) ** 2) ** 0.5
                            _gate_pass = (_mahal <= _HU_MAHAL_ANISO)

                    if _HU_GATE_DEBUG:
                        print(f"[门·主] W{self.window_id} f{self.frame_index} "
                              f"det#{c_idx}→ID{tid} du={du:.4f}/{u_thresh:.3f} "
                              f"dv={dv:.4f}/{v_thresh:.3f} "
                              f"{'过' if _gate_pass else '拒'}")

                    if _gate_pass:
                        dist_uv = (du ** 2 + dv ** 2) ** 0.5
                        _diag = _diag_ref
                        _SIDE_STAT["pos_pass"] += 1
                        if _MATCH_MODE == "sidegate" and _HU_METRIC == "shape_add":
                            # 附加门：位置门已过，再要求 shape 分达标；不达标则拒（不改变打分）
                            _sh = self._ciou_motion(
                                du, dv, dist_uv, _diag,
                                c_head.get('rw', 0.0),
                                p_info.get('rw', c_head.get('rw', 0.0)),
                                c_head.get('rh', 0.0),
                                p_info.get('rh', c_head.get('rh', 0.0)))
                            if _sh < _HU_SHAPE_GATE:
                                _SIDE_STAT["gate_reject"] += 1
                                continue
                            _sim = self._max_sim(tid, c_head.get('appearance'))
                            if _sim is not None and _sim < _APP_GATE:
                                _SIDE_STAT["gate_reject"] += 1
                                continue
                            _motion = 1.0 - min(1.0, dist_uv / _diag)
                            if _sim is None:
                                _score = _motion
                            else:
                                _score = _WM_MOT * _motion + (1.0 - _WM_MOT) * _sim
                            candidates.append((-float(_score), c_idx, tid))
                        elif _MATCH_MODE == "sidegate" and _HU_METRIC in ("shape", "shape_nopos"):
                            # 主门：shape 分（面积比 + 长宽比）
                            _motion = self._ciou_motion(
                                du, dv, dist_uv, _diag,
                                c_head.get('rw', 0.0),
                                p_info.get('rw', c_head.get('rw', 0.0)),
                                c_head.get('rh', 0.0),
                                p_info.get('rh', c_head.get('rh', 0.0)))
                            if _motion < _HU_SHAPE_GATE:
                                _SIDE_STAT["gate_reject"] += 1
                                continue
                            _sim = self._max_sim(tid, c_head.get('appearance'))
                            if _sim is not None and _sim < _APP_GATE:
                                _SIDE_STAT["gate_reject"] += 1
                                continue
                            # shape_nopos：打分只用 shape + 微弱位置兜底（避免完全无位置信息）
                            if _HU_METRIC == "shape_nopos":
                                _pos = 1.0 - min(1.0, dist_uv / max(1.0, 3.0))
                                candidates.append((-float(_motion), c_idx, tid))
                            else:
                                candidates.append((-float(_motion), c_idx, tid))
                        elif _MATCH_MODE == "sidegate":
                            # sideview 式双门控：位置已过门限，外观再做硬 AND 门（sim>=_APP_GATE 才准入），
                            # 缺一不可；之后门内按 运动×外观 打分（外观主导，w_motion=_WM_MOT）。
                            # 与 hybrid 的差别：hybrid 只把外观当门内软加权，外观低也可能位置匹配。
                            _sim = self._max_sim(tid, c_head.get('appearance'))
                            if _sim is not None:
                                _SIDE_STAT["app_checked"] += 1
                                _SIDE_SIMS.append(float(_sim))
                                if _sim < _APP_GATE:
                                    _SIDE_STAT["app_reject"] += 1
                                    continue
                            if _HU_METRIC == "ciou":
                                # CIoU 式：中心距离 + 尺度一致性 + 宽高比（纯几何，不依赖外观）
                                _motion = self._ciou_motion(
                                    du, dv, dist_uv, _diag,
                                    c_head.get('rw', 0.0),
                                    p_info.get('rw', c_head.get('rw', 0.0)),
                                    c_head.get('rh', 0.0),
                                    p_info.get('rh', c_head.get('rh', 0.0)))
                            else:
                                _motion = 1.0 - min(1.0, dist_uv / _diag)
                            if _sim is None:
                                _score = _motion
                            else:
                                _score = _WM_MOT * _motion + (1.0 - _WM_MOT) * _sim
                            if _FUSE_W is not None:
                                # 三路融合：把"尺度+长宽比"作为【独立第三项】加入打分
                                # （此前 HU_METRIC=ciou 是【替换】运动项，二者不同）
                                _sh3 = self._ciou_motion(
                                    du, dv, dist_uv, _diag,
                                    c_head.get('rw', 0.0),
                                    p_info.get('rw', c_head.get('rw', 0.0)),
                                    c_head.get('rh', 0.0),
                                    p_info.get('rh', c_head.get('rh', 0.0)))
                                _s_app = 0.0 if _sim is None else max(0.0, float(_sim))
                                _w = _FUSE_W
                                _den = _w[0] + _w[1] + _w[2]
                                if _den <= 1e-9:
                                    _den = 1.0
                                _score = (_w[0] * _motion + _w[1] * _s_app
                                          + _w[2] * _sh3) / _den
                            candidates.append((-float(_score), c_idx, tid))
                            if _HU_APP_LOGIC or _HU_TRACE:
                                _pair_sim[(c_idx, tid)] = _sim
                                _pair_mot[(c_idx, tid)] = _motion
                        elif _MATCH_MODE == "hybrid":
                            # 门内加权（对比基准）：外观只当门内软权项，不是硬门槛。
                            _motion = 1.0 - min(1.0, dist_uv / _diag)
                            _sim = None
                            _pa = p_info.get('appearance')
                            if c_head.get('appearance') is not None and _pa is not None:
                                _sim = float(np.clip(
                                    float(np.dot(c_head['appearance'], _pa)), 0.0, 1.0))
                            if _sim is None:
                                _score = _motion
                            else:
                                _score = _HYBRID_WM * _motion + (1.0 - _HYBRID_WM) * _sim
                            candidates.append((-float(_score), c_idx, tid))
                        else:
                            size_delta = (
                                abs(c_head['rw'] - p_info.get('rw', c_head['rw'])) +
                                abs(c_head['rh'] - p_info.get('rh', c_head['rh']))
                            )
                            candidates.append((dist_uv + 0.35 * size_delta, c_idx, tid))

            if handoff_blocked_tracks:
                # Once one strong handoff is seen, the old identity must not
                # fall back to a second-best detection in the same frame.
                candidates = [item for item in candidates
                              if int(item[2]) not in handoff_blocked_tracks]
            candidates.sort(key=lambda x: x[0])

            # ============ HU 外观用法改造：相对判据 + 互选（默认关闭）============
            # 绝对阈值把"外观能否区分"当成全局常数；本块改为在每帧候选集合内部做
            # 相对判断：外观能区分时由外观拍板（剪掉次要竞争边），不能区分时外观
            # 弃权、该检测退回运动打分。判据随本帧尺度自适应，与特征尺度解耦。
            if _HU_APP_LOGIC and candidates:
                _best_det, _best_trk, _dsims = {}, {}, {}
                for _k, _ci, _tid in candidates:
                    _s = _pair_sim.get((_ci, _tid))
                    _s = -2.0 if _s is None else _s
                    _dsims.setdefault(_ci, []).append(_s)
                    if _ci not in _best_det or _s > _best_det[_ci][0]:
                        _best_det[_ci] = (_s, _tid)
                    if _tid not in _best_trk or _s > _best_trk[_tid][0]:
                        _best_trk[_tid] = (_s, _ci)
                _kept = []
                for _k, _ci, _tid in candidates:
                    _s = _pair_sim.get((_ci, _tid))
                    _s = -2.0 if _s is None else _s
                    _bs, _btid = _best_det[_ci]
                    _cands = sorted(_dsims[_ci], reverse=True)
                    _second = _cands[1] if len(_cands) > 1 else None
                    _clear = (
                        _tid == _btid and (
                            _second is None or
                            (_bs - _second) >= _HU_REL_MARGIN or
                            _bs >= _HU_REL_ABS
                        )
                    ) if _HU_REL_MARGIN is not None else True
                    if _HU_REL_MARGIN is not None:
                        if _tid == _btid and _clear:
                            pass                       # 外观拍板：保留最优轨
                        elif _tid != _btid:
                            _SIDE_STAT["rel_prune"] += 1
                            continue                   # 剪掉次要竞争边
                        else:
                            # 外观无法区分（最优也领先不足）→ 外观弃权，退回运动
                            _k = -float(_pair_mot.get((_ci, _tid), 0.0))
                    if _HU_MNN and (_btid != _tid or _best_trk[_tid][1] != _ci):
                        _SIDE_STAT["mnn_prune"] += 1
                        continue
                    _kept.append((_k, _ci, _tid))
                candidates = sorted(_kept, key=lambda x: x[0])
            # ============ HU 外观用法改造结束 ============

            # ================= HU 原型改动（相对原脚本唯一行为差异）=================
            # 原版：按"单对最高分"贪心指派 —— 两轨两检测时会选出交叉配对（实测
            #   8 例拆/并成对，如 20260808120053537 帧5 id2/id4 完全互换）。
            # 本版：对门内候选 (检测 × 轨道) 代价矩阵做全局最优指派
            #   目标 = 配对数最多，其次总代价最小；小规模 bitmask DP，
            #   ≤12×12 精确，超出退化为原贪心。不新增任何阈值/先验。
            def _assign(_cands):
                _pc = {}
                for _key, _ci, _tid in _cands:
                    _pc.setdefault((_ci, _tid), _key)
                if _HU_TRACE:
                    _hu_pairs.extend([(int(ci), int(ti), round(float(k), 4),
                                       None if _pair_sim.get((ci, ti)) is None
                                       else round(float(_pair_sim[(ci, ti)]), 4),
                                       round(float(_pair_mot.get((ci, ti), 0.0)), 4))
                                      for (ci, ti), k in _pc.items()])
                _dets_u = sorted({ci for _, ci, _ in _cands})
                _trs_u = sorted({tid for _, _, tid in _cands})
                if _dets_u and _trs_u and len(_dets_u) <= 12 and len(_trs_u) <= 12:
                    _dp = {0: (0, 0.0)}          # mask -> (-配对数, 总代价)
                    _choices = []
                    for _ci in _dets_u:
                        _ndp = {}
                        _ch = {}
                        for _mask, (_np, _cs) in _dp.items():
                            if _mask not in _ndp or (_np, _cs) < _ndp[_mask]:
                                _ndp[_mask] = (_np, _cs)
                                _ch[_mask] = -1
                            for _k, _tid in enumerate(_trs_u):
                                if _mask >> _k & 1:
                                    continue
                                _c = _pc.get((_ci, _tid))
                                if _c is None:
                                    continue
                                _val = (_np - 1, _cs + _c)
                                _nm = _mask | (1 << _k)
                                if _nm not in _ndp or _val < _ndp[_nm]:
                                    _ndp[_nm] = _val
                                    _ch[_nm] = _k
                        _dp = _ndp
                        _choices.append(_ch)
                    _best_mask = min(_dp, key=lambda m: _dp[m])
                    _mask = _best_mask
                    _assign = []
                    for _i in range(len(_dets_u) - 1, -1, -1):
                        _k = _choices[_i].get(_mask, -1)
                        if _k >= 0:
                            _assign.append((_dets_u[_i], _trs_u[_k]))
                            _mask ^= (1 << _k)
                    for _ci, _tid in _assign:
                        if _ci in matched_curr or _tid in matched_prev_ids:
                            continue
                        matched_curr.add(_ci)
                        if _HU_TRACE:
                            _hu_matched.append((int(_ci), int(_tid)))
                        matched_prev_ids.add(_tid)
                        update_track(_tid, curr_heads[_ci])
                else:
                    for dist_uv, c_idx, tid in _cands:
                        if c_idx not in matched_curr and tid not in matched_prev_ids:
                            matched_curr.add(c_idx)
                            matched_prev_ids.add(tid)
                            c_head = curr_heads[c_idx]
                            update_track(tid, c_head)

            if _HU_TWOPASS > 0:
                # 第 1 趟：只用高分检测（= 基线行为）
                _hi = [(k, ci, tid) for (k, ci, tid) in candidates
                       if curr_heads[ci].get('conf') is None
                       or float(curr_heads[ci]['conf']) >= _HU_TWOPASS]
                _assign(_hi)
                # 第 2 趟：低分检测只对第 1 趟没配上的轨道指派
                _lo = [(k, ci, tid) for (k, ci, tid) in candidates
                       if curr_heads[ci].get('conf') is not None
                       and float(curr_heads[ci]['conf']) < _HU_TWOPASS
                       and ci not in matched_curr and tid not in matched_prev_ids]
                _assign(_lo)
            else:
                _assign(candidates)
            # ================= HU 原型改动结束 =================

        # FIX-20260827-V2: dual-coordinate recovery for changing/clipped windows.
        #
        # Normal matching above uses pane-relative (u, v), which is excellent
        # while the detected window geometry is stable.  When a window changes
        # width strongly or reaches an image boundary, however, u/v can jump
        # even though the passenger follows a smooth trajectory in image space.
        #
        # Before creating a new ID, re-associate unmatched detections using:
        #   1) absolute image anchor (cx, cy) + per-track image velocity,
        #   2) window/global motion_shift as a fallback motion prior,
        #   3) vertical pane position and detector-box scale as safeguards.
        #
        # This is intended to reconnect cases such as ID202 -> ID204 caused by
        # a changing Window 2 box, without imposing a hard max-person cap.
        if is_window_unstable or is_window_clipped or _HU_V2_ALWAYS:
            unmatched_curr_indices = [
                idx for idx in range(num_curr) if idx not in matched_curr
            ]

            # Stable old IDs are preferred. A one-frame tentative fragment may
            # not steal a detection from a confirmed passenger.
            recovery_track_ids = [
                tid for tid in track_ids
                if (
                    tid not in matched_prev_ids and
                    tid in self.unique_head_ids and
                    self.tracked_heads[tid].get('hits', 0) >= 2 and
                    self.tracked_heads[tid].get('missed_frames', 99) <= 2
                )
            ]

            recovery_candidates = []
            window_dx, window_dy = motion_shift if motion_shift is not None else (0.0, 0.0)

            for c_idx in unmatched_curr_indices:
                c_head = curr_heads[c_idx]

                for tid in recovery_track_ids:
                    p_info = self.tracked_heads[tid]
                    elapsed = max(1, p_info.get('missed_frames', 1))

                    # Absolute head motion learned from previous matches.  For a
                    # newly-confirmed track, blend toward the measured window
                    # translation because its own velocity estimate is short.
                    head_vx = float(p_info.get('dcx_velocity', 0.0))
                    head_vy = float(p_info.get('dcy_velocity', 0.0))
                    if abs(head_vx) < 2.0:
                        pred_vx = float(window_dx)
                    else:
                        pred_vx = 0.75 * head_vx + 0.25 * float(window_dx)
                    if abs(head_vy) < 2.0:
                        pred_vy = float(window_dy)
                    else:
                        pred_vy = 0.75 * head_vy + 0.25 * float(window_dy)

                    pred_cx = float(p_info['cx']) + pred_vx * elapsed
                    pred_cy = float(p_info['cy']) + pred_vy * elapsed
                    dx_px = abs(float(c_head['cx']) - pred_cx)
                    dy_px = abs(float(c_head['cy']) - pred_cy)

                    # Scale pixel residuals by the current pane size.  Keep a
                    # pixel floor because a heavily clipped pane can be narrow.
                    x_gate_px = max(85.0, 0.42 * win_w) + 25.0 * max(0, elapsed - 1)
                    y_gate_px = max(55.0, 0.50 * win_h) + 15.0 * max(0, elapsed - 1)

                    # v is usually much less damaged by horizontal clipping than
                    # u, so retain it as a soft consistency check.
                    pred_v = (
                        float(p_info['v']) +
                        float(p_info.get('dv_velocity', 0.0)) * elapsed
                    )
                    dv_norm = abs(float(c_head['v']) - pred_v)

                    curr_w = max(1.0, float(c_head['w']))
                    curr_h = max(1.0, float(c_head['h']))
                    prev_w = max(1.0, float(p_info.get('w', curr_w)))
                    prev_h = max(1.0, float(p_info.get('h', curr_h)))
                    width_sim = min(curr_w, prev_w) / max(curr_w, prev_w)
                    height_sim = min(curr_h, prev_h) / max(curr_h, prev_h)

                    # Appearance is only a tie-breaker.  Lighting/viewpoint
                    # changes must never by themselves force a new identity.
                    app_distance = None
                    if (c_head.get('appearance') is not None and
                            p_info.get('appearance') is not None):
                        app_distance = float(1.0 - np.clip(
                            np.dot(c_head['appearance'], p_info['appearance']),
                            -1.0, 1.0))

                    if _HU_GATE_DEBUG:
                        print(f"[门·V2] W{self.window_id} f{self.frame_index} "
                              f"det#{c_idx}→ID{tid} dv={dv_norm:.4f}/{_V2_MAX_DV:.3f} "
                              f"dx={dx_px:.1f}/{x_gate_px:.1f} dy={dy_px:.1f}/"
                              f"{y_gate_px:.1f} "
                              f"{'过' if (dx_px <= x_gate_px and dy_px <= y_gate_px and dv_norm <= _V2_MAX_DV and width_sim >= 0.30 and height_sim >= 0.30) else '拒'}")

                    # 方案 A：V2 同样要过各向异性门
                    if _HU_MAHAL_ANISO > 0:
                        _sv2 = p_info.get('resid_ema_v')
                        if _sv2 is not None:
                            if dv_norm > _HU_MAHAL_ANISO * max(
                                    _HU_MAHAL_FLOOR_V, float(_sv2)):
                                continue

                    if (dx_px <= x_gate_px and
                            dy_px <= y_gate_px and
                            dv_norm <= _V2_MAX_DV and
                            width_sim >= 0.30 and
                            height_sim >= 0.30):
                        cost = (
                            dx_px / x_gate_px +
                            0.80 * dy_px / y_gate_px +
                            0.45 * dv_norm +
                            0.25 * (1.0 - width_sim) +
                            0.25 * (1.0 - height_sim) +
                            0.12 * p_info.get('missed_frames', 0)
                        )
                        if app_distance is not None:
                            cost += 0.08 * min(app_distance, 1.0)
                        recovery_candidates.append((cost, c_idx, tid))

            # One-to-one greedy assignment over the strongest absolute-motion
            # candidates. Track counts are tiny per pane, so this is stable and
            # avoids adding a scipy dependency on RK3588.
            recovery_candidates.sort(key=lambda item: item[0])
            for _, c_idx, tid in recovery_candidates:
                if c_idx in matched_curr or tid in matched_prev_ids:
                    continue

                c_head = curr_heads[c_idx]
                # Save normalized-u velocity because the current window
                # coordinate system is unstable; update_track may otherwise
                # learn the artificial u jump as real passenger motion.
                old_du_velocity = self.tracked_heads[tid].get('du_velocity', 0.0)
                update_track(tid, c_head)
                self.tracked_heads[tid]['du_velocity'] = old_du_velocity
                if tid in self.track_summaries:
                    self.track_summaries[tid]['du_velocity'] = old_du_velocity

                matched_curr.add(c_idx)
                matched_prev_ids.add(tid)

        # ============================================================
        # FIX-20260827-V7: history re-association BEFORE creating NEW IDs
        #
        # A confirmed passenger may disappear for 1-2 frames while the
        # physical window keeps moving/scaling.  Comparing the old raw u/v
        # against the new raw u/v can fail because the window coordinate
        # system itself has changed.  Before any unmatched detection is
        # allowed to become a NEW identity, re-project each recently-lost
        # confirmed track's last pane-relative position into the CURRENT
        # window box:
        #
        #     projected_cx = win_x1 + old_u * current_win_w
        #     projected_cy = win_y1 + old_v * current_win_h
        #
        # Then perform a one-to-one minimum-cost assignment using projected
        # image position, vertical consistency, box-size continuity and
        # appearance only as a weak tie-breaker.
        #
        # This directly handles patterns such as:
        #   F3-F5: ID201 stable
        #   F6:    missed
        #   F7:    two detections appear
        # One of the F7 detections should inherit ID201; only the remaining
        # detection is a genuinely new physical passenger.
        # ============================================================
        unmatched_curr_indices = [
            idx for idx in range(num_curr)
            if idx not in matched_curr
        ]
        recent_lost_confirmed_ids = [
            tid for tid in track_ids
            if (
                tid not in matched_prev_ids
                and tid in self.unique_head_ids
                and tid in self.tracked_heads
                and 1 <= int(self.tracked_heads[tid].get('missed_frames', 0)) <= 2
                and int(self.tracked_heads[tid].get('hits', 0)) >= 2
            )
        ]

        history_reassoc_candidates = []

        for tid in recent_lost_confirmed_ids:
            p_info = self.tracked_heads[tid]
            missed = max(1, int(p_info.get('missed_frames', 1)))

            # Reproject the old pane-relative anchor into the current window.
            projected_cx = win_x1 + float(p_info['u']) * win_w
            projected_cy = win_y1 + float(p_info['v']) * win_h

            # Also keep a weak absolute-motion prediction.  This is useful
            # when the pane geometry is stable, but must not dominate when
            # the window detector changes scale.
            abs_pred_cx = float(p_info.get('cx', projected_cx)) + \
                float(p_info.get('dcx_velocity', 0.0)) * missed
            abs_pred_cy = float(p_info.get('cy', projected_cy)) + \
                float(p_info.get('dcy_velocity', 0.0)) * missed

            for c_idx in unmatched_curr_indices:
                c_head = curr_heads[c_idx]

                reproj_dx = abs(float(c_head['cx']) - projected_cx) / max(1.0, win_w)
                reproj_dy = abs(float(c_head['cy']) - projected_cy) / max(1.0, win_h)
                reproj_dist = (reproj_dx * reproj_dx + reproj_dy * reproj_dy) ** 0.5

                abs_dx = abs(float(c_head['cx']) - abs_pred_cx) / max(1.0, win_w)
                abs_dy = abs(float(c_head['cy']) - abs_pred_cy) / max(1.0, win_h)
                abs_dist = (abs_dx * abs_dx + abs_dy * abs_dy) ** 0.5

                dv = abs(float(c_head['v']) - float(p_info['v']))

                prev_w = max(1.0, float(p_info.get('w', c_head['w'])))
                prev_h = max(1.0, float(p_info.get('h', c_head['h'])))
                curr_w = max(1.0, float(c_head['w']))
                curr_h = max(1.0, float(c_head['h']))
                width_sim = min(prev_w, curr_w) / max(prev_w, curr_w)
                height_sim = min(prev_h, curr_h) / max(prev_h, curr_h)

                app_distance = None
                if (p_info.get('appearance') is not None and
                        c_head.get('appearance') is not None):
                    app_distance = float(1.0 - np.clip(
                        np.dot(p_info['appearance'], c_head['appearance']),
                        -1.0, 1.0))

                # Gate mainly on the window-reprojected anchor.  The gate is
                # intentionally moderate because occupant position within a
                # detected pane can change as the pane is clipped/perspective
                # warped, but it still rejects the clearly farther second
                # passenger in a two-detection frame.
                if _HU_GATE_DEBUG:
                    print(f"[门·V7] W{self.window_id} f{self.frame_index} "
                          f"det#{c_idx}→ID{tid} dv={dv:.4f}/{_V7_MAX_DV:.3f} "
                          f"reproj={reproj_dist:.4f}/0.380 reproj_dy={reproj_dy:.4f}/0.300 "
                          f"{'过' if (reproj_dist <= 0.38 and reproj_dy <= 0.30 and dv <= _V7_MAX_DV and width_sim >= 0.35 and height_sim >= 0.35) else '拒'}")

                # 方案 A：V7 也要过各向异性门，否则主门拒绝了、V7 又用松门接回来
                if _HU_MAHAL_ANISO > 0:
                    _sv7 = p_info.get('resid_ema_v')
                    if _sv7 is not None:
                        if dv > _HU_MAHAL_ANISO * max(_HU_MAHAL_FLOOR_V,
                                                      float(_sv7)):
                            continue

                if (reproj_dist <= 0.38 and
                        reproj_dy <= 0.30 and
                        dv <= _V7_MAX_DV and
                        width_sim >= 0.35 and
                        height_sim >= 0.35):
                    cost = (
                        1.35 * reproj_dist +
                        0.35 * min(abs_dist, 1.0) +
                        0.35 * dv +
                        0.20 * (1.0 - width_sim) +
                        0.20 * (1.0 - height_sim) +
                        0.06 * (missed - 1)
                    )
                    if app_distance is not None:
                        cost += 0.06 * min(app_distance, 1.0)

                    history_reassoc_candidates.append((
                        cost, c_idx, tid,
                        reproj_dist, reproj_dx, reproj_dy, abs_dist
                    ))

        history_reassoc_candidates.sort(key=lambda item: item[0])
        used_history_tracks = set()

        for (cost, c_idx, tid, reproj_dist, reproj_dx,
             reproj_dy, abs_dist) in history_reassoc_candidates:
            if c_idx in matched_curr or tid in matched_prev_ids or tid in used_history_tracks:
                continue
            if int(tid) in handoff_blocked_tracks:
                continue
            # HU 原型实验开关：V7 低证据接受上限（默认关闭）
            if _HU_V7_COST_MAX and float(cost) > float(_HU_V7_COST_MAX):
                continue

            c_head = curr_heads[c_idx]
            old_du_velocity = self.tracked_heads[tid].get('du_velocity', 0.0)
            old_missed_frames = int(
                self.tracked_heads[tid].get('missed_frames', 0)
            )

            # V19: 先在 update_track() 重置 missed_frames 之前判断这次
            # 历史重关联是否已经“过于勉强”。
            #
            # 这种结构：
            #   A(ID201) -> LOST -> D(仍被匹配成ID201)
            # 如果 cost、窗口重投影误差和绝对运动误差都很大，
            # 就不应再把它仅作为普通断轨恢复；计数层要把前后两段
            # 当作两个物理身份。
            strong_history_identity_break = (
                old_missed_frames >= 1 and
                float(cost) >= 0.75 and
                float(reproj_dist) >= 0.25 and
                float(abs_dist) >= 0.45
            )

            if _R1_SPLIT_ON_BREAK and strong_history_identity_break:
                # R-1（2026-09-14）：本次重关联已被判为"过于勉强"（V19 证据），
                # 却仍把检测并入旧 tid → 一个 gid 覆盖两个物理人 → 少算。
                # 这里放弃该配对，让检测走 认领 / 新建 id 路径。
                # 必须与 _CLAIM_THR=0.65 同时开：放开的检测会被认领路径接回
                # 【正确的过期 gid】，而不是变成新 id（单独开 R-1 会误伤）。
                continue

            # V67: update_track() 前保留本次历史重关联的质量指标。
            # old_missed_frames>=1 已由 history reassociation 路径保证；
            # 最终层再结合实际 observation 与稳定 anchor 做跨侧判断。
            self.history_reassoc_records.append({
                'track_id': int(tid),
                'frame': int(self.frame_index),
                'missed': int(old_missed_frames),
                'cost': float(cost),
                'reproj': float(reproj_dist),
                'abs': float(abs_dist),
            })

            update_track(tid, c_head)

            if strong_history_identity_break:
                self.history_reassoc_identity_breaks.add(
                    (int(tid), int(self.frame_index))
                )
                print(
                    f"[历史重关联身份断裂] W{self.window_id} | "
                    f"ID{tid} | frame={self.frame_index} | "
                    f"missed={old_missed_frames} | "
                    f"cost={cost:.3f} | "
                    f"reproj={reproj_dist:.3f} | "
                    f"abs={abs_dist:.3f}"
                )

            # The re-association itself may occur after a pane-coordinate jump;
            # keep the prior normalized horizontal velocity instead of learning
            # that jump as real occupant motion.
            self.tracked_heads[tid]['du_velocity'] = old_du_velocity
            if tid in self.track_summaries:
                self.track_summaries[tid]['du_velocity'] = old_du_velocity

            matched_curr.add(c_idx)
            matched_prev_ids.add(tid)
            used_history_tracks.add(tid)

            print(
                f"[历史轨迹重关联] W{self.window_id} | "
                f"ID{tid} <- det#{c_idx} | "
                f"frame={self.frame_index} | "
                f"cost={cost:.3f} | "
                f"reproj={reproj_dist:.3f} "
                f"(dx={reproj_dx:.3f},dy={reproj_dy:.3f}) | "
                f"abs={abs_dist:.3f}"
            )

        # —— sideview 阶段二：全局外观认领（在本帧开新 id 之前，把漏配的检测认领回"已过期 gid"）——
        if _MATCH_MODE == "sidegate" and _ENABLE_CLAIM:
            claimed_gids = set()
            for c_idx in range(num_curr):
                if c_idx in matched_curr:
                    continue
                c_head = curr_heads[c_idx]
                gid = self._claim_expired_identity(
                    c_head, c_idx, curr_heads, matched_curr, matched_prev_ids)
                if gid is None or gid in claimed_gids:
                    continue
                claimed_gids.add(gid)
                # 重建同 gid 轨迹（身份复活，不再开新 id → distinct 不再虚增）
                self.tracked_heads[gid] = {
                    'u': c_head['u'], 'v': c_head['v'],
                    'cx': c_head['cx'], 'cy': c_head['cy'],
                    'w': c_head['w'], 'h': c_head['h'],
                    'rw': c_head['rw'], 'rh': c_head['rh'],
                    'du_velocity': 0.0, 'dv_velocity': 0.0,
                    'dcx_velocity': 0.0, 'dcy_velocity': 0.0,
                    'hits': 1,
                    'ttl': min(3, self.max_ttl), 'missed_frames': 0
                }
                if c_head.get('appearance') is not None:
                    self.tracked_heads[gid]['appearance'] = c_head['appearance']
                    self._archive_appearance(gid, c_head['appearance'])
                sm = self.track_summaries[gid]
                sm['last_frame'] = self.frame_index
                sm['last_u'] = c_head['u']; sm['last_v'] = c_head['v']
                sm['last_cx'] = c_head['cx']; sm['last_cy'] = c_head['cy']
                sm['last_rw'] = c_head['rw']; sm['last_rh'] = c_head['rh']
                sm['frames'].add(self.frame_index)
                sm['observations'][self.frame_index] = {
                    'u': c_head['u'], 'v': c_head['v'],
                    'rw': c_head['rw'], 'rh': c_head['rh'],
                    'cx': c_head['cx'], 'cy': c_head['cy'],
                    'w': c_head['w'], 'h': c_head['h'],
                    'conf': c_head.get('conf'),
                    'person_prob': c_head.get('person_prob')
                }
                if c_head.get('appearance') is not None:
                    sm.setdefault('appearances', {})[self.frame_index] = c_head['appearance']
                    sm['last_appearance'] = c_head['appearance']
                matched_curr.add(c_idx)
                _SIDE_STAT["claim"] += 1
                print(f"[sideview认领] W{self.window_id} | det#{c_idx} -> ID{gid} @F{self.frame_index}")

        if _HU_OCC:
            # 帧末定稿采集：每个 gid 本帧的真实观测（一次一个）。
            # obs_prev 在 V7 认领前取，避免把"本帧刚被认领/新建的 gid"混进观测。
            self._occ_clean_obs = getattr(self, '_occ_clean_obs', {})
            for _tid in set(list(track_ids) + list(matched_prev_ids)):
                _o = (self.track_summaries.get(int(_tid)) or {}).get(
                    'observations', {}).get(self.frame_index)
                if _o is not None:
                    self._occ_clean_obs.setdefault(int(_tid), []).append(
                        (int(self.frame_index), float(_o['u']), float(_o['v'])))
            for _tid, _lst in _occ_obs.items():
                self._occ_clean_obs.setdefault(int(_tid), []).extend(_lst)

        for c_idx in range(num_curr):
            if c_idx not in matched_curr:
                c_head = curr_heads[c_idx]

                # HU_LOWCONF_CREATE_MIN：低分框不得建新轨（ByteTrack 式安全属性）。
                # 匹配不上即丢弃 —— 低分假阳因此无法进入 track_summaries 被永久计数，
                # 低分真人只能去延长已有轨道，不会凭空多出一个身份。
                if (_HU_LOWCONF_CREATE_MIN > 0 and c_head.get('conf') is not None
                        and float(c_head['conf']) < _HU_LOWCONF_CREATE_MIN):
                    _SIDE_STAT["lowconf_drop"] += 1
                    continue

                new_id = self.window_id * 100 + self.next_head_id
                self.next_head_id += 1
                if _HU_TRACE:
                    _hu_newids.append((int(c_idx), int(new_id),
                                       round(c_head['u'], 3), round(c_head['v'], 3)))
                
                self.tracked_heads[new_id] = {
                    'u': c_head['u'], 'v': c_head['v'],
                    'cx': c_head['cx'], 'cy': c_head['cy'], 
                    'w': c_head['w'], 'h': c_head['h'],
                    'rw': c_head['rw'], 'rh': c_head['rh'],
                    'du_velocity': 0.0, 'dv_velocity': 0.0,
                    'dcx_velocity': 0.0, 'dcy_velocity': 0.0,
                    'hits': 1,
                    'ttl': min(3, self.max_ttl), 'missed_frames': 0
                }
                if c_head.get('appearance') is not None:
                    self.tracked_heads[new_id]['appearance'] = c_head['appearance']
                    self._archive_appearance(new_id, c_head['appearance'])
                self.track_summaries[new_id] = {
                    'first_frame': self.frame_index,
                    'last_frame': self.frame_index,
                    'first_u': c_head['u'], 'first_v': c_head['v'],
                    'last_u': c_head['u'], 'last_v': c_head['v'],
                    'first_rw': c_head['rw'], 'first_rh': c_head['rh'],
                    'last_rw': c_head['rw'], 'last_rh': c_head['rh'],
                    'du_velocity': 0.0, 'dv_velocity': 0.0,
                    'first_cx': c_head['cx'], 'first_cy': c_head['cy'],
                    'last_cx': c_head['cx'], 'last_cy': c_head['cy'],
                    'dcx_velocity': 0.0, 'dcy_velocity': 0.0,
                    'first_appearance': c_head.get('appearance'),
                    'last_appearance': c_head.get('appearance'),
                    'appearances': ({self.frame_index: c_head.get('appearance')}
                                    if c_head.get('appearance') is not None else {}),
                    'frames': {self.frame_index},
                    'observations': {
                        self.frame_index: {
                            'u': c_head['u'], 'v': c_head['v'],
                            'rw': c_head['rw'], 'rh': c_head['rh'],
                            'cx': c_head['cx'], 'cy': c_head['cy'],
                            'w': c_head['w'], 'h': c_head['h'],
                            'conf': c_head.get('conf'),
                            'person_prob': c_head.get('person_prob')
                        }
                    }
                }
                if c_idx in verified_handoff_curr:
                    self.verified_handoff_ids.add(int(new_id))
                    for old_tid in verified_handoff_curr[c_idx]:
                        self.cannot_merge_pairs.add(
                            frozenset((int(old_tid), int(new_id))))

                # A third rear passenger can be visible for only one frame at
                # the vehicle exit.  Accept that singleton only when the rear
                # window has already proved that two people coexist, two
                # stable tracks already exist, the window geometry is stable,
                # and normal motion-gated matching rejected this detection.
                # The final max_detected + 1 guard still caps the contribution
                # to one extra passenger, preventing runaway over-counting.
                confirmed_active = [
                    tid for tid in track_ids
                    if tid in self.unique_head_ids
                ]
                confirmed_pred_distances = []
                for confirmed_tid in confirmed_active:
                    confirmed_info = self.tracked_heads.get(confirmed_tid)
                    if confirmed_info is None:
                        continue
                    confirmed_elapsed = max(
                        1, confirmed_info.get('missed_frames', 1)
                    )
                    confirmed_pred_u = (
                        confirmed_info['u'] +
                        confirmed_info.get('du_velocity', 0.0) * confirmed_elapsed
                    )
                    confirmed_pred_v = (
                        confirmed_info['v'] +
                        confirmed_info.get('dv_velocity', 0.0) * confirmed_elapsed
                    )
                    confirmed_pred_distances.append((
                        (c_head['u'] - confirmed_pred_u) ** 2 +
                        (c_head['v'] - confirmed_pred_v) ** 2
                    ) ** 0.5)
                clearly_distinct_from_confirmed = (
                    bool(confirmed_pred_distances) and
                    min(confirmed_pred_distances) >= 0.18
                )
                previous_max = max(self.frame_detected_counts[:-1], default=0)
                allow_rear_exit_singleton = (
                    self.window_id != 1 and
                    previous_max >= 2 and
                    len(confirmed_active) >= 2 and
                    self.frame_index >= 4 and
                    # FIX-20260827: never promote a one-frame NEW identity while
                    # the pane is clipped.  Its normalized u coordinate is not
                    # reliable enough to prove a fourth passenger.
                    not is_window_clipped and
                    (not is_window_unstable or clearly_distinct_from_confirmed)
                )
                other_curr_distances = [
                    ((c_head['u'] - other_head['u']) ** 2 +
                     (c_head['v'] - other_head['v']) ** 2) ** 0.5
                    for other_idx, other_head in enumerate(curr_heads)
                    if other_idx != c_idx
                ]
                other_curr_ious = [
                    iou(c_head['box'], other_head['box'])
                    for other_idx, other_head in enumerate(curr_heads)
                    if other_idx != c_idx
                ]
                allow_same_frame_distinct = (
                    self.window_id != 1 and
                    num_curr >= 2 and
                    bool(other_curr_distances) and
                    (
                        min(other_curr_distances) >= 0.12 or
                        (bool(other_curr_ious) and max(other_curr_ious) <= 0.35)
                    ) and
                    not is_window_unstable and
                    # FIX-20260827: same-frame spatial separation in normalized
                    # window coordinates is not trustworthy at an image edge.
                    not is_window_clipped
                )
                if allow_rear_exit_singleton or allow_same_frame_distinct:
                    self.unique_head_ids.add(new_id)

        # In a rear pane, two passengers may only become visible one at a
        # time.  Once one track is stable, retroactively accept one earlier
        # single-frame track only when it is far away in normalized window
        # coordinates and has remained unmatched for multiple frames.
        if (self.window_id != 1 and
                max(self.frame_detected_counts, default=0) == 1 and
                self.unique_head_ids):
            confirmed_infos = [
                self.tracked_heads[tid]
                for tid in self.unique_head_ids
                if tid in self.tracked_heads
            ]
            if confirmed_infos:
                for tentative_tid, tentative_info in self.tracked_heads.items():
                    if tentative_tid in self.unique_head_ids:
                        continue
                    if (tentative_info.get('hits', 1) != 1 or
                            tentative_info.get('missed_frames', 0) < 2):
                        continue
                    nearest_confirmed = min(
                        (
                            (tentative_info['u'] - confirmed['u']) ** 2 +
                            (tentative_info['v'] - confirmed['v']) ** 2
                        ) ** 0.5
                        for confirmed in confirmed_infos
                    )
                    if nearest_confirmed >= 0.25:
                        self.unique_head_ids.add(tentative_tid)
                        break

        # V40: 当前帧所有 ID 分配完成后，永久记录同帧共存关系。
        self._mark_coexisting_ids(self.frame_index)

        if _HU_TRACE:
            _log = getattr(self, '_hu_trace_log', None)
            if _log is None:
                _log = []
                self._hu_trace_log = _log
            _log.append({'f': self.frame_index, 'w': self.window_id, 'dets': _hu_dets,
                         'tracks_pre': _hu_tracks_pre, 'pairs': _hu_pairs,
                         'matched': _hu_matched, 'new_ids': _hu_newids})

        return self.count_distinct()


    def count_distinct(self):
        """该窗跨帧出现过的不同 gid 数（无先验的裸 distinct 口径）。
        HU_OCC=1 时改用"占用约束"口径：把观测轨迹含大跳变的 gid 按段拆开计数。"""
        if not _HU_OCC:
            return len(self.track_summaries)
        return len(self._occ_segments())

    def _occ_segments(self):
        """把每个 gid 按"共现证据"切成若干物理身份段，返回段标识集合。

        证据链（位置 + 帧共现，不依赖外观阈值）：
          · 交接帧的特征是"旧人还在观测、新人同时出现"，两者被并进同一 gid；
          · 该 gid 在同一帧因此出现两个观测（或与另一 gid 的近距观测同帧），
            而位置门/共现约束证明这是两个物理人 —— 一个 gid 只能是一个人。
          · 于是：一个 gid 的存活期内，每出现一次"同帧多观测 / 与另一 gid 同帧近距"
            的冲突，就在该帧处切一刀，段数 = 该 gid 实际容纳的人数。

        观测不足或从未冲突的 gid 退化为单段（保持裸 distinct 的原有判定）。
        """
        obs_by_gid = self._occ_clean_obs or {}
        frames = {}
        for tid, lst in obs_by_gid.items():
            for (f, u, v) in lst:
                frames.setdefault(int(f), []).append((int(tid), float(u), float(v)))
        segs = set()
        for tid in self.track_summaries:
            obs = sorted({(int(f), round(float(u), 4), round(float(v), 4))
                          for (f, u, v) in obs_by_gid.get(int(tid), [])})
            if not obs:
                segs.add(f"{int(tid)}#0")
                continue
            seg = 0
            for (f, u, v) in obs:
                # 同帧是否有第二个物理人与本 gid 冲突（同 gid 多观测，或另一 gid 近距同帧）
                conflict = False
                others = [o for o in frames.get(f, []) if o[0] != int(tid)]
                same_gid_dup = sum(1 for o in obs if o[0] == f
                                   and (o[1], o[2]) != (u, v))
                if same_gid_dup:
                    conflict = True
                else:
                    for (_o_tid, ou, ov) in others:
                        d = ((ou - u) ** 2 + (ov - v) ** 2) ** 0.5
                        if d <= _HU_OCC_COEXIST:
                            conflict = True
                            break
                if conflict:
                    seg += 1
                    _SIDE_STAT["occ_split"] += 1
            # 段数 = 冲突次数 + 1（首个冲突前的观测属于第一段）
            segs.add(f"{int(tid)}#{seg}")
        return segs

    def occ_diagnostics(self):
        """HU_OCC 诊断：列出被切段的 gid 及其共现冲突帧。"""
        obs_by_gid = self._occ_clean_obs or {}
        frames = {}
        for tid, lst in obs_by_gid.items():
            for (f, u, v) in lst:
                frames.setdefault(int(f), []).append((int(tid), float(u), float(v)))
        out = []
        for tid in sorted(self.track_summaries):
            obs = sorted({(int(f), round(float(u), 4), round(float(v), 4))
                          for (f, u, v) in obs_by_gid.get(int(tid), [])})
            hits = []
            for (f, u, v) in obs:
                dup = sum(1 for o in obs if o[0] == f and (o[1], o[2]) != (u, v))
                near = [(o[0], round(((o[1]-u)**2 + (o[2]-v)**2) ** 0.5, 3))
                        for o in frames.get(f, []) if o[0] != int(tid)
                        and ((o[1]-u)**2 + (o[2]-v)**2) ** 0.5 <= _HU_OCC_COEXIST]
                if dup or near:
                    hits.append({"f": f, "same_gid_dup": dup,
                                 "near_other_gid": near})
            if hits:
                out.append({"tid": int(tid), "n_obs": len(obs),
                            "conflicts": hits, "n_seg": len(hits) + 1})
        return out

# ★ 移植块结束



class _DGWindowTrackerAdapter:
    """把移植版 WindowHeadTracker 包成新脚本的 tracker 接口。

    计数口径 = 裸 distinct（`count_distinct()` = `len(track_summaries)`），
    **不含**原 get_count() 的轨迹缝合与 V 块补人 —— 那部分经实测为
    缝合净贡献 0、补人靠场景拟合，且夹带座位先验，故不采用。
    """

    def __init__(self, wid):
        self._t = WindowHeadTracker(window_id=wid)
        self._swin = None          # 方案 B：窗框纵向坐标的时域平滑状态
        self.last_suppressed_track_ids = set()

    def update(self, dets, feats, confs, person_probs=None, motion_shift=None,
               window_box=None, img_w=1920, is_head_on_left=True):
        # 方案 B：只平滑窗口框的 y1/y3（纵向），x 保持检测原值
        if _HU_WIN_SMOOTH > 0 and window_box is not None:
            wb = list(window_box)
            if self._swin is None:
                self._swin = [wb[1], wb[3]]
            else:
                a = min(0.95, max(0.0, _HU_WIN_SMOOTH))
                self._swin[0] = a * self._swin[0] + (1.0 - a) * wb[1]
                self._swin[1] = a * self._swin[1] + (1.0 - a) * wb[3]
            if self._swin[1] - self._swin[0] > 4.0:
                wb[1], wb[3] = self._swin[0], self._swin[1]
                window_box = tuple(wb)
        self._t.update(
            dets,
            window_box=window_box,
            motion_shift=(0.0, 0.0) if motion_shift is None else motion_shift,
            is_head_on_left=is_head_on_left,
            img_w=img_w,
            appearance_features=feats,
            head_confs=confs,
            person_probs=person_probs,
        )
        # 每帧刷新物理身份并查集，使重连后的新内部 ID 在标注层沿用旧 ID。
        self.general_count()
        # 该 tracker 不返回逐头标签，只有聚合计数 → 标签位留空
        return [None] * len(dets)

    def total_count(self):
        return int(self._t.count_distinct())

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.size != b.size or not a.size:
            return None
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / den) if den > 1e-8 else None

    @staticmethod
    def _adaptive_boundary(t, f0, f1, old_obs, new_obs):
        """Return motion/geometry evidence for an identity boundary.

        A boundary is sparse when the vehicle travels a large fraction of a
        pane between observations, regardless of the total image count.
        Prediction residual is measured in head-box units in image space.
        """
        shifts = [t.frame_motion_shifts.get(f, (0.0, 0.0))
                  for f in range(int(f0) + 1, int(f1) + 1)]
        pred_dx = sum(float(s[0]) for s in shifts)
        pred_dy = sum(float(s[1]) for s in shifts)
        actual_dx = float(new_obs["cx"] - old_obs["cx"])
        actual_dy = float(new_obs["cy"] - old_obs["cy"])
        head_scale = max(1.0, float(old_obs.get("w", 1.0)),
                         float(old_obs.get("h", 1.0)),
                         float(new_obs.get("w", 1.0)),
                         float(new_obs.get("h", 1.0)))
        prediction_residual = float(np.hypot(
            actual_dx - pred_dx, actual_dy - pred_dy)) / head_scale
        qualities = [t.frame_observation_quality.get(f, {})
                     for f in range(int(f0) + 1, int(f1) + 1)]
        motion_ratio = max((float(q.get("motion_ratio", 0.0))
                            for q in qualities), default=0.0)
        geometry_reliable = bool(qualities) and not any(
            bool(q.get("unstable", True)) for q in qualities)
        # A missed detection does not by itself make the sampling sparse: when
        # per-frame vehicle motion is small, accumulated motion prediction is
        # still informative across the gap. Sparsity follows physical motion.
        sparse_observation = motion_ratio >= 0.20
        return {
            "prediction_residual": prediction_residual,
            "motion_ratio": motion_ratio,
            "geometry_reliable": geometry_reliable,
            "sparse": sparse_observation,
        }

    def general_count(self):
        """无座位先验的保守人数估计。

        峰值是硬下限；稳定轨迹默认各算一人；单帧轨迹只有在与稳定轨迹
        同帧且空间分离时才晋升。互不共存、时间相邻且位置/外观连续的轨迹
        视作同一人的碎片。任何同帧共存证据都禁止合并。
        """
        t = self._t
        summaries = t.track_summaries
        rejected_observations = set()
        rejected_track_ids = set()
        person_threshold = None  # 不使用 OSNet 真假人分类分数
        self.last_suppressed_track_ids = set()
        stable = {
            int(tid) for tid, sm in summaries.items()
            if (len(sm.get("frames", ())) >= GENERAL_MIN_HITS and
                int(tid) not in rejected_track_ids)
        }
        # A duplicate box can appear beside a mature person late in the
        # sequence, receive a fresh ID, and survive for exactly two frames.
        # Prefer the already-established identity even when the duplicate box
        # happens to have higher detector confidence.  This lifecycle proof is
        # intentionally narrow and does not change the fixed hit requirement.
        shadow_newcomers = set()
        for tid in list(stable):
            sm = summaries[tid]
            frames = sorted(int(f) for f in sm.get("frames", ()))
            if not frames or len(frames) > 2:
                continue
            first_frame = frames[0]
            obs = sm.get("observations", {}).get(first_frame)
            if obs is None:
                continue
            for peer_id, peer_sm in summaries.items():
                peer_id = int(peer_id)
                if peer_id == tid or peer_id in rejected_track_ids:
                    continue
                peer_frames = sorted(int(f) for f in peer_sm.get("frames", ()))
                if (first_frame not in peer_frames or
                        sum(f < first_frame for f in peer_frames) < 2):
                    continue
                peer_obs = peer_sm.get("observations", {}).get(first_frame)
                if peer_obs is None:
                    continue
                distance = float(np.hypot(float(obs["cx"]) - float(peer_obs["cx"]),
                                          float(obs["cy"]) - float(peer_obs["cy"])))
                min_width = max(1.0, min(float(obs["w"]), float(peer_obs["w"])))
                y_overlap = max(
                    0.0,
                    min(float(obs["cy"]) + float(obs["h"]) * 0.5,
                        float(peer_obs["cy"]) + float(peer_obs["h"]) * 0.5)
                    - max(float(obs["cy"]) - float(obs["h"]) * 0.5,
                          float(peer_obs["cy"]) - float(peer_obs["h"]) * 0.5))
                y_ratio = y_overlap / max(
                    1.0, min(float(obs["h"]), float(peer_obs["h"])))
                x_overlap = max(
                    0.0,
                    min(float(obs["cx"]) + float(obs["w"]) * 0.5,
                        float(peer_obs["cx"]) + float(peer_obs["w"]) * 0.5)
                    - max(float(obs["cx"]) - float(obs["w"]) * 0.5,
                          float(peer_obs["cx"]) - float(peer_obs["w"]) * 0.5))
                x_ratio = x_overlap / max(
                    1.0, min(float(obs["w"]), float(peer_obs["w"])))
                area_a = float(obs["w"] * obs["h"])
                area_b = float(peer_obs["w"] * peer_obs["h"])
                size_ratio = min(area_a, area_b) / max(1.0, area_a, area_b)
                if (distance <= 0.85 * min_width and x_ratio >= 0.55 and
                        y_ratio >= 0.82 and
                        size_ratio >= 0.65):
                    shadow_newcomers.add(tid)
                    if GENERAL_COUNT_TRACE:
                        print(f"[成熟轨迹旁影子ID过滤] W{t.window_id} "
                              f"drop=ID{tid} keep=ID{peer_id} "
                              f"frame={first_frame} dist={distance:.1f} "
                              f"x={x_ratio:.2f} y={y_ratio:.2f} "
                              f"size={size_ratio:.2f}")
                    break
        if shadow_newcomers:
            rejected_track_ids.update(shadow_newcomers)
            stable.difference_update(shadow_newcomers)
            self.last_suppressed_track_ids = set(rejected_track_ids)
        # A persistent false body/seat box may occupy an ID for two weak frames,
        # then inherit an unrelated far-away detection during a window-number
        # transition.  Do not let that final high-confidence jump turn the
        # contaminated ID into a valid person.
        # Two weak sightings separated by a long gap are typical reflections
        # that were re-linked across the moving vehicle, not a persistent head.
        sparse_weak_tracks = set()
        for tid in list(stable):
            sm = summaries[tid]
            frames = sorted(int(f) for f in sm.get("frames", ()))
            obs_by_frame = sm.get("observations", {})
            confs = [float(obs_by_frame[f]["conf"]) for f in frames
                     if f in obs_by_frame and obs_by_frame[f].get("conf") is not None]
            if (len(frames) == 2 and frames[1] - frames[0] >= 4 and
                    len(confs) == 2 and max(confs) < 0.40):
                sparse_weak_tracks.add(int(tid))
                self.last_suppressed_track_ids.add(int(tid))
                if GENERAL_COUNT_TRACE:
                    print(f"[稀疏弱误检轨迹] W{t.window_id} ID{tid} "
                          f"frames={frames} confs={[round(x, 3) for x in confs]}")
        stable.difference_update(sparse_weak_tracks)

        contaminated_weak_start = set()
        for tid in list(stable):
            sm = summaries[tid]
            frames = sorted(int(f) for f in sm.get("frames", ()))
            obs_by_frame = sm.get("observations", {})
            if len(frames) != 3 or frames[1] != frames[0] + 1:
                continue
            first_obs = [obs_by_frame.get(frames[0]), obs_by_frame.get(frames[1])]
            last_obs = obs_by_frame.get(frames[2])
            if any(o is None or o.get("conf") is None for o in first_obs) or \
                    last_obs is None or last_obs.get("conf") is None:
                continue
            weak_start = max(float(o["conf"]) for o in first_obs) <= 0.30
            strong_finish = float(last_obs["conf"]) >= 0.60
            u_jump = abs(float(last_obs["u"]) - float(first_obs[-1]["u"]))
            strong_coexisting_peers = 0
            for peer_id, peer_sm in summaries.items():
                if int(peer_id) == int(tid):
                    continue
                peer_obs = peer_sm.get("observations", {})
                if all(f in peer_obs and peer_obs[f].get("conf") is not None and
                       float(peer_obs[f]["conf"]) >= 0.45
                       for f in frames[:2]):
                    strong_coexisting_peers += 1
            if (weak_start and strong_finish and u_jump >= 0.18 and
                    strong_coexisting_peers >= 2):
                contaminated_weak_start.add(int(tid))
                if GENERAL_COUNT_TRACE:
                    print(f"[弱误检轨迹污染] W{t.window_id} ID{tid} "
                          f"weak={[round(float(o['conf']), 3) for o in first_obs]} "
                          f"finish={float(last_obs['conf']):.3f} u_jump={u_jump:.3f}")
        stable.difference_update(contaminated_weak_start)
        promoted = set()
        peak_promoted = set()
        separated_only_promoted = set()
        forced_separate_pairs = set()
        temporal_crowded_singletons = set()
        for tid, sm in summaries.items():
            tid = int(tid)
            frames = set(sm.get("frames", ()))
            if tid in stable or len(frames) != 1:
                continue
            f = next(iter(frames))
            obs = sm.get("observations", {}).get(f)
            if obs is None:
                continue
            verified_late_window_override = (
                int(t.window_id) >= 3 and len(summaries) == 1 and
                max(t.frame_detected_counts, default=0) == 1 and
                obs.get("conf") is not None and float(obs["conf"]) >= 0.28 and
                person_threshold is not None and obs.get("person_prob") is not None and
                float(obs["person_prob"]) >= max(0.50, float(person_threshold) * 3.0) and
                0.08 <= float(obs["u"]) <= 0.92)
            verified_handoff_override = (
                int(tid) in getattr(t, "verified_handoff_ids", set()))
            same_frame_stable_ids = [
                int(sid) for sid in stable
                if f in summaries[sid].get("observations", {})]
            temporal_crowded_override = (
                len(stable) >= 3 and len(same_frame_stable_ids) >= 2 and
                len(same_frame_stable_ids) < len(stable) and
                obs.get("conf") is not None and float(obs["conf"]) >= 0.15 and
                person_threshold is not None and
                obs.get("person_prob") is not None and
                float(obs["person_prob"]) >= float(person_threshold) and
                1 <= int(f) <= len(t.frame_detected_counts) and
                int(t.frame_detected_counts[int(f) - 1]) >= 3 and
                all(frozenset((int(tid), sid)) in t.cannot_merge_pairs
                    for sid in same_frame_stable_ids))
            if temporal_crowded_override:
                temporal_crowded_singletons.add(int(tid))
                rejected_track_ids.discard(tid)
                self.last_suppressed_track_ids.discard(tid)
            if verified_handoff_override:
                rejected_track_ids.discard(tid)
                self.last_suppressed_track_ids.discard(tid)
            if (tid in rejected_track_ids and
                    not verified_late_window_override and
                    not verified_handoff_override and
                    not temporal_crowded_override):
                continue
            singleton_person_cutoff = (float(person_threshold) * GENERAL_SINGLETON_PERSON_FACTOR
                                       if person_threshold is not None else None)
            if (singleton_person_cutoff is not None and int(t.window_id) >= 3 and
                    len(summaries) == 1):
                singleton_person_cutoff = max(0.50, singleton_person_cutoff)
            if (not verified_late_window_override and
                    not verified_handoff_override and
                    not temporal_crowded_override and
                    person_threshold is not None and
                    obs.get("person_prob") is not None and
                    float(obs["person_prob"]) <
                    singleton_person_cutoff):
                if GENERAL_COUNT_TRACE:
                    print(f"[OSNet单帧晋升过滤] W{t.window_id} ID{tid} "
                          f"prob={float(obs['person_prob']):.6f} "
                          f"cutoff={float(person_threshold) * GENERAL_SINGLETON_PERSON_FACTOR:.6f}")
                # Deferred rendering must hide the same final false track.
                self.last_suppressed_track_ids.add(int(tid))
                continue
            stable_distances = []
            stable_peers = []
            duplicate_of_stable = False
            for sid in stable:
                other = summaries[sid].get("observations", {}).get(f)
                if other is None:
                    continue
                d = float(np.hypot(obs["u"] - other["u"], obs["v"] - other["v"]))
                stable_distances.append(d)
                stable_peers.append((d, other))
                pixel_dist = float(np.hypot(
                    obs["cx"] - other["cx"], obs["cy"] - other["cy"]))
                y_overlap = max(
                    0.0,
                    min(obs["cy"] + obs["h"] * 0.5,
                        other["cy"] + other["h"] * 0.5)
                    - max(obs["cy"] - obs["h"] * 0.5,
                          other["cy"] - other["h"] * 0.5)
                ) / max(1.0, min(float(obs["h"]), float(other["h"])))
                size_ratio = min(float(obs["w"] * obs["h"]),
                                 float(other["w"] * other["h"])) / max(
                                     1.0, float(max(obs["w"] * obs["h"],
                                                    other["w"] * other["h"])))
                app_sim = self._cosine(
                    sm.get("appearances", {}).get(f),
                    summaries[sid].get("appearances", {}).get(f))
                if (pixel_dist <= GENERAL_DUP_CENTER * max(float(obs["w"]), float(other["w"]))
                        and y_overlap >= GENERAL_DUP_Y_OVERLAP
                        and size_ratio >= GENERAL_DUP_SIZE_RATIO
                        and app_sim is not None and app_sim >= GENERAL_DUP_APP):
                    duplicate_of_stable = True
            strong_crowded_singleton = False
            if len(stable_peers) >= 2 and obs.get("conf") is not None:
                _, nearest = min(stable_peers, key=lambda item: item[0])
                peer_conf = nearest.get("conf")
                if peer_conf is not None:
                    conf = float(obs["conf"])
                    conf_ratio = min(conf, float(peer_conf)) / max(
                        1e-6, conf, float(peer_conf))
                    strong_crowded_singleton = (
                        conf >= GENERAL_SINGLETON_STRONG_CONF and
                        conf_ratio >= GENERAL_SINGLETON_PEER_CONF_RATIO)
            coexisting_all_stable = (
                len(stable_peers) == len(stable) and len(stable) >= 1)
            peak_supported_singleton = (
                coexisting_all_stable and obs.get("conf") is not None and
                (float(obs["conf"]) >= GENERAL_SINGLETON_PEAK_CONF or
                 (float(obs["conf"]) >= 0.40 and stable_distances and
                  min(stable_distances) >= 0.16)) and
                max(t.frame_detected_counts, default=0) >= len(stable) + 1)
            # In a genuinely crowded frame (four or more simultaneous heads),
            # a lower-confidence occupant can be tightly occluded by neighbours.
            # Preserve it when it coexists with every established trajectory;
            # this still requires an actual four-head frame and does not relax
            # the normal two/three-frame stability thresholds.
            dense_peak_supported_singleton = (
                coexisting_all_stable and obs.get("conf") is not None and
                float(obs["conf"]) >= 0.30 and
                1 <= int(f) <= len(t.frame_detected_counts) and
                int(t.frame_detected_counts[int(f) - 1]) >= 4 and
                max(t.frame_detected_counts, default=0) >= 4)
            # A rear/edge window can remain visible for several frames while a
            # heavily occluded occupant is detected only once.  Accept that
            # observation only when the window history contains no competing
            # track at all, the detection is away from the window border, and
            # the window has supplied at least three temporal observations.
            # This is window-level temporal support, not a relaxation of the
            # ordinary per-track two/three-frame requirements.
            isolated_window_singleton = (
                len(summaries) == 1 and
                max(t.frame_detected_counts, default=0) == 1 and
                len(t.frame_detected_counts) >= 3 and
                obs.get("conf") is not None and
                float(obs["conf"]) >= 0.30 and
                0.12 <= float(obs["u"]) <= 0.88 and
                0.08 <= float(obs["v"]) <= 0.92)
            # A window can enter only in the final frame of an 8-frame sequence.
            # Admit its sole detection only with strong OSNet person evidence;
            # this keeps the normal hit-count rules unchanged for ordinary windows.
            late_window_verified_singleton = verified_late_window_override
            crowded_three_singleton = False
            if (len(stable) == 2 and len(stable_peers) == 2 and
                    obs.get("conf") is not None and
                    1 <= int(f) <= len(t.frame_detected_counts) and
                    int(t.frame_detected_counts[int(f) - 1]) >= 3 and
                    max(t.frame_detected_counts, default=0) >= 3):
                _, nearest = min(stable_peers, key=lambda item: item[0])
                nearest_conf = nearest.get("conf")
                if nearest_conf is not None:
                    singleton_conf = float(obs["conf"])
                    nearest_ratio = min(singleton_conf, float(nearest_conf)) / max(
                        1e-6, singleton_conf, float(nearest_conf))
                    crowded_three_singleton = (
                        singleton_conf >= 0.18 and nearest_ratio >= 0.45)
            # Sometimes a complete four-person window is visible for only one
            # frame, so no stable anchor exists.  Promote the whole high-
            # confidence peak cluster; isolated tracks in lower-occupancy frames
            # are deliberately excluded.
            crowded_four_peak_singleton = (
                len(frames) == 1 and obs.get("conf") is not None and
                1 <= int(f) <= len(t.frame_detected_counts) and
                int(t.frame_detected_counts[int(f) - 1]) >= 4 and
                int(t.frame_detected_counts[int(f) - 1]) ==
                max(t.frame_detected_counts, default=0) and
                float(obs["conf"]) >= 0.40)
            temporal_distinct_singleton = False
            if not stable_distances and obs.get("conf") is not None:
                for sid in stable:
                    sframes = sorted(int(x) for x in summaries[sid].get("frames", ()))
                    if not sframes:
                        continue
                    nearest_gap = min(abs(f - sf) for sf in sframes)
                    sim = self._cosine(
                        sm.get("last_appearance"),
                        summaries[sid].get("first_appearance"))
                    adjacent_distinct = (
                        nearest_gap == 1 and
                        float(obs["conf"]) >= GENERAL_TEMPORAL_SINGLETON_CONF and
                        sim is not None and sim < GENERAL_STITCH_APP)
                    # Historical 8-frame sequences can miss the intermediate
                    # view of an occupant.  Keep the fixed hit requirement,
                    # but allow a one-frame identity only when a two-frame gap
                    # is backed by all three independent signals: adequate
                    # detector confidence, a large normalized seat-position
                    # separation, and clearly different OSNet appearance.
                    short_gap_distinct = (
                        len(t.frame_detected_counts) <= 8 and nearest_gap == 2 and
                        float(obs["conf"]) >= GENERAL_SHORT_SINGLETON_CONF and
                        abs(float(obs["u"]) - float(
                            summaries[sid].get("first_u", obs["u"]))) >=
                        GENERAL_SHORT_SINGLETON_DU and
                        sim is not None and sim < GENERAL_SHORT_SINGLETON_APP)
                    if adjacent_distinct or short_gap_distinct:
                        temporal_distinct_singleton = True
                        forced_separate_pairs.add(frozenset((tid, int(sid))))
                        if short_gap_distinct and GENERAL_COUNT_TRACE:
                            print(f"[短序列两帧间隔真人证据] W{t.window_id} "
                                  f"ID{tid}/ID{sid} frame={f} gap={nearest_gap} "
                                  f"du={abs(float(obs['u']) - float(summaries[sid].get('first_u', obs['u']))):.3f} "
                                  f"sim={sim:.3f} conf={float(obs['conf']):.3f} "
                                  f"prob={float(obs.get('person_prob') or 0.0):.6f}")
            # 单帧轨迹必须与同帧所有稳定轨迹都分离才算新人；只要贴近
            # 任一稳定轨迹，就更可能是该人的重复检测框。
            if ((not duplicate_of_stable and stable_distances
                    and min(stable_distances) >= GENERAL_SINGLETON_SEP)
                    or strong_crowded_singleton
                    or peak_supported_singleton
                    or dense_peak_supported_singleton
                    or verified_handoff_override
                    or temporal_crowded_override
                    or isolated_window_singleton
                    or late_window_verified_singleton
                    or crowded_three_singleton
                    or crowded_four_peak_singleton
                    or temporal_distinct_singleton):
                promoted.add(tid)
                separated_rule = (not duplicate_of_stable and bool(stable_distances)
                                  and min(stable_distances) >= GENERAL_SINGLETON_SEP)
                supported_rule = any((
                    strong_crowded_singleton, peak_supported_singleton,
                    dense_peak_supported_singleton, verified_handoff_override,
                    temporal_crowded_override, isolated_window_singleton,
                    late_window_verified_singleton, crowded_three_singleton,
                    crowded_four_peak_singleton, temporal_distinct_singleton))
                if separated_rule and not supported_rule:
                    separated_only_promoted.add(int(tid))
                if GENERAL_COUNT_TRACE:
                    reasons = []
                    for reason_name, reason_value in (
                            ("separated", not duplicate_of_stable and bool(stable_distances) and min(stable_distances) >= GENERAL_SINGLETON_SEP),
                            ("strong_crowded", strong_crowded_singleton),
                            ("peak", peak_supported_singleton),
                            ("dense_peak", dense_peak_supported_singleton),
                            ("handoff", verified_handoff_override),
                            ("temporal_crowded", temporal_crowded_override),
                            ("isolated", isolated_window_singleton),
                            ("late", late_window_verified_singleton),
                            ("crowded3", crowded_three_singleton),
                            ("crowded4", crowded_four_peak_singleton),
                            ("temporal_distinct", temporal_distinct_singleton)):
                        if reason_value:
                            reasons.append(reason_name)
                    print(f"[adaptive singleton promotion] W{t.window_id} "
                          f"ID{tid} F{f} reasons={reasons}")
                if (peak_supported_singleton or dense_peak_supported_singleton
                        or crowded_four_peak_singleton):
                    peak_promoted.add(tid)
            elif GENERAL_COUNT_TRACE and not stable_distances:
                for sid in stable:
                    sframes = sorted(int(x) for x in summaries[sid].get("frames", ()))
                    if sframes and min(abs(f - sf) for sf in sframes) <= 2:
                        sim = self._cosine(
                            sm.get("last_appearance"),
                            summaries[sid].get("first_appearance"))
                        print(f"[时序单帧诊断] W{t.window_id} ID{tid}/ID{sid} "
                              f"frame={f} gap={min(abs(f - sf) for sf in sframes)} "
                              f"du={abs(float(obs['u']) - float(summaries[sid].get('first_u', obs['u']))):.3f} "
                              f"sim={sim if sim is not None else -1:.3f}")

        # Two genuinely different occupants can both be visible for only one
        # frame. Keep the normal hit rule, but accept simultaneous pair-level
        # evidence when both tracks passed OSNet and duplicate suppression.
        eligible_singletons = []
        for pair_tid, pair_sm in summaries.items():
            pair_tid = int(pair_tid)
            pair_frames = sorted(int(x) for x in pair_sm.get("frames", ()))
            if (len(pair_frames) == 1 and
                    pair_tid not in rejected_track_ids and
                    pair_tid not in sparse_weak_tracks):
                pair_obs = pair_sm.get("observations", {}).get(pair_frames[0])
                if pair_obs is not None:
                    eligible_singletons.append(
                        (pair_tid, pair_frames[0], pair_obs, pair_sm))
        if (len(eligible_singletons) == 2 and
                eligible_singletons[0][1] == eligible_singletons[1][1]):
            left_item, right_item = eligible_singletons
            pair_frame = left_item[1]
            left_obs, right_obs = left_item[2], right_item[2]
            frame_peak = (int(t.frame_detected_counts[pair_frame - 1])
                          if 1 <= pair_frame <= len(t.frame_detected_counts)
                          else 0)
            x_sep = abs(float(left_obs["cx"]) - float(right_obs["cx"]))
            min_width = max(1.0, min(float(left_obs["w"]),
                                     float(right_obs["w"])))
            sep_ratio = x_sep / min_width
            pair_sim = self._cosine(
                left_item[3].get("appearances", {}).get(pair_frame),
                right_item[3].get("appearances", {}).get(pair_frame))
            pair_confs = [float(left_obs.get("conf") or 0.0),
                          float(right_obs.get("conf") or 0.0)]
            distinct_pair = (sep_ratio >= 0.75 or
                             (pair_sim is not None and pair_sim < 0.78))
            paired_single_frame_people = (
                frame_peak >= 2 and max(t.frame_detected_counts, default=0) >= 2 and
                sum(int(c) > 0 for c in t.frame_detected_counts) == 1 and
                max(pair_confs) >= 0.65 and min(pair_confs) >= 0.12 and
                distinct_pair)
            if paired_single_frame_people:
                promoted.update((left_item[0], right_item[0]))
                if GENERAL_COUNT_TRACE:
                    print(f"[same-frame person pair] W{t.window_id} "
                          f"IDs={left_item[0]},{right_item[0]} F{pair_frame} "
                          f"peak={frame_peak} sep={sep_ratio:.3f} "
                          f"app={pair_sim if pair_sim is not None else -1:.3f} "
                          f"conf={pair_confs}")

        # If stable identities already explain the simultaneous occupancy peak,
        # a separated one-frame box may add a sequential newcomer only when the
        # boundary remains observable after vehicle-motion compensation. Fast
        # motion or a changing pane makes u/v separation alone insufficient.
        observed_raw_peak = int(max(t.frame_detected_counts, default=0))
        for weak_tid in list(separated_only_promoted):
            if len(stable) < observed_raw_peak:
                continue
            weak_sm = summaries.get(weak_tid, {})
            weak_frames = sorted(int(x) for x in weak_sm.get("frames", ()))
            if len(weak_frames) != 1:
                continue
            wf = weak_frames[0]
            wo = weak_sm.get("observations", {}).get(wf)
            boundary_candidates = []
            for sid in stable:
                so_all = summaries[sid].get("observations", {})
                for sf in sorted(int(x) for x in so_all):
                    if sf == wf:
                        continue
                    gap = abs(wf - sf)
                    if gap > 2:
                        continue
                    old_f, new_f = (sf, wf) if sf < wf else (wf, sf)
                    old_o, new_o = ((so_all[sf], wo) if sf < wf
                                    else (wo, so_all[sf]))
                    ev = self._adaptive_boundary(t, old_f, new_f, old_o, new_o)
                    sim = self._cosine(
                        summaries[sid].get("appearances", {}).get(sf),
                        weak_sm.get("appearances", {}).get(wf))
                    boundary_candidates.append((gap, ev, sim, int(sid)))
            boundary_candidates.sort(key=lambda item: item[0])
            # A high-confidence singleton that directly coexists with a stable,
            # spatially separate identity is a distinct person observation.
            # Keep it even if a previous track has similar appearance: with
            # sparse views and weak imaging, slot separation is stronger evidence.
            coexisting_separated = []
            for sid in stable:
                peer_obs = summaries[sid].get("observations", {}).get(wf)
                if peer_obs is None or frozenset((int(weak_tid), int(sid))) not in t.cannot_merge_pairs:
                    continue
                slot_sep = float(np.hypot(
                    float(wo["u"]) - float(peer_obs["u"]),
                    float(wo["v"]) - float(peer_obs["v"])))
                if slot_sep >= GENERAL_SINGLETON_SEP:
                    coexisting_separated.append((int(sid), slot_sep))
            high_conf_coexisting = (
                wo.get("conf") is not None and
                float(wo["conf"]) >= GENERAL_SINGLETON_PEAK_CONF and
                bool(coexisting_separated))
            verified_newcomer = high_conf_coexisting
            if high_conf_coexisting and GENERAL_COUNT_TRACE:
                print(f"[高置信同帧共存单人保留] W{t.window_id} ID{weak_tid} "
                      f"F{wf} conf={float(wo['conf']):.3f} peers={coexisting_separated}")
            for _, ev, sim, _ in boundary_candidates:
                required_residual = 0.85 if ev["sparse"] else 0.60
                if (ev["geometry_reliable"] and
                        ev["prediction_residual"] >= required_residual and
                        sim is not None and sim < 0.72):
                    verified_newcomer = True
                    break
            if not verified_newcomer:
                promoted.discard(weak_tid)
                if GENERAL_COUNT_TRACE:
                    print(f"[adaptive singleton veto] W{t.window_id} ID{weak_tid} "
                          f"F{wf} stable={len(stable)} peak={observed_raw_peak}")

        # If stable trajectories already explain the full observed occupancy
        # peak, a very weak singleton cannot add an extra simultaneous seat.
        # Such boxes are commonly reflections at entry/exit or oversized edge
        # crops.  This does not alter the fixed hit requirement and is applied
        # only when at least two stable people already account for the peak.
        if len(stable) >= 2 and observed_raw_peak == len(stable):
            redundant_weak_singletons = set()
            for tid in promoted:
                if int(tid) in temporal_crowded_singletons:
                    continue
                sm = summaries.get(tid, {})
                frames = list(sm.get("frames", ()))
                if len(frames) != 1:
                    continue
                f = int(frames[0])
                obs = sm.get("observations", {}).get(f)
                # Preserve the disappearing member of a verified three-track
                # boundary swap: another stable ID spans f->f+1 and a distinct
                # stable newcomer starts at f+1.  This is the sample-41 pattern
                # and represents three physical people despite a two-head peak.
                boundary_swap_evidence = False
                if obs is not None:
                    for new_sid in stable:
                        new_sm = summaries[new_sid]
                        new_frames = sorted(int(x) for x in new_sm.get("frames", ()))
                        if not new_frames or new_frames[0] != f + 1:
                            continue
                        new_obs = new_sm.get("observations", {}).get(f + 1)
                        new_sim = self._cosine(
                            sm.get("appearances", {}).get(f),
                            new_sm.get("appearances", {}).get(f + 1))
                        if new_obs is None or new_sim is None:
                            continue
                        new_uv = float(np.hypot(float(new_obs["u"]) - float(obs["u"]),
                                                float(new_obs["v"]) - float(obs["v"])))
                        has_boundary_bridge = any(
                            bridge_sid not in (tid, new_sid) and
                            f in summaries[bridge_sid].get("observations", {}) and
                            f + 1 in summaries[bridge_sid].get("observations", {})
                            for bridge_sid in stable)
                        if has_boundary_bridge and new_uv >= 0.25 and new_sim < 0.65:
                            boundary_swap_evidence = True
                            break
                if (obs is not None and float(obs.get("conf", 0.0)) < 0.30 and
                        not boundary_swap_evidence):
                    redundant_weak_singletons.add(int(tid))
            if redundant_weak_singletons:
                promoted.difference_update(redundant_weak_singletons)
                peak_promoted.difference_update(redundant_weak_singletons)
                if GENERAL_COUNT_TRACE:
                    print(f"[稳定峰值弱单帧过滤] W{t.window_id} "
                          f"ids={sorted(redundant_weak_singletons)} "
                          f"stable={len(stable)} peak={observed_raw_peak}")

        # A fast vehicle can move far enough that the same isolated occupant is
        # assigned a fresh ID in every frame.  Preserve the normal per-track hit
        # requirement, but aggregate a chain only when the window peak is one,
        # every member is a singleton, frames are consecutive, and adjacent
        # appearances agree for at least three frames.
        forced_singleton_links = []
        if max(t.frame_detected_counts, default=0) == 1:
            singletons_by_frame = {}
            for tid, sm in summaries.items():
                if int(tid) in rejected_track_ids:
                    continue
                fs = sorted(int(f) for f in sm.get("frames", ()))
                if len(fs) == 1:
                    singletons_by_frame.setdefault(fs[0], []).append(int(tid))
            chain = []
            for f in sorted(singletons_by_frame):
                tids_at_f = singletons_by_frame[f]
                if len(tids_at_f) != 1:
                    chain = []
                    continue
                tid = tids_at_f[0]
                if chain:
                    prev_tid, prev_f = chain[-1]
                    sim = self._cosine(
                        summaries[prev_tid].get("last_appearance"),
                        summaries[tid].get("first_appearance"))
                    if f != prev_f + 1 or sim is None or sim < GENERAL_SINGLETON_CHAIN_APP:
                        chain = []
                chain.append((tid, f))
                if len(chain) >= GENERAL_SINGLETON_CHAIN_MIN:
                    chain_ids = [item[0] for item in chain]
                    promoted.update(chain_ids)
                    forced_singleton_links.extend(
                        zip(chain_ids[:-1], chain_ids[1:]))
            # 同一个人被中间一帧遮挡时，可能形成两个单帧碎轨。
            # 仅当整个窗峰值为 1、恰好只有这两个碎轨，且位置/外观
            # 都高度连续时重连；合并后总观测数仍是 2，不改两帧要求。
            _single_ids = [tid for tids in singletons_by_frame.values() for tid in tids]
            if len(_single_ids) == 2:
                _left_id, _right_id = _single_ids
                _left, _right = summaries[_left_id], summaries[_right_id]
                _lf = next(iter(_left.get("frames", ())))
                _rf = next(iter(_right.get("frames", ())))
                if _lf > _rf:
                    _left_id, _right_id = _right_id, _left_id
                    _left, _right, _lf, _rf = _right, _left, _rf, _lf
                _sim = self._cosine(
                    _left.get("last_appearance"), _right.get("first_appearance"))
                _uv = float(np.hypot(
                    float(_right.get("first_u", 0.0) - _left.get("last_u", 0.0)),
                    float(_right.get("first_v", 0.0) - _left.get("last_v", 0.0))))
                _left_obs = _left.get("observations", {}).get(_lf, {})
                _right_obs = _right.get("observations", {}).get(_rf, {})
                _pair_min_conf = min(
                    float(_left_obs.get("conf") or 0.0),
                    float(_right_obs.get("conf") or 0.0))
                _pair_min_person = min(
                    float(_left_obs.get("person_prob") or 0.0),
                    float(_right_obs.get("person_prob") or 0.0))
                _pair_size_ratio = max(
                    float(_left_obs.get("w") or 1.0) * float(_left_obs.get("h") or 1.0),
                    float(_right_obs.get("w") or 1.0) * float(_right_obs.get("h") or 1.0)) / max(
                        1.0,
                        min(float(_left_obs.get("w") or 1.0) * float(_left_obs.get("h") or 1.0),
                            float(_right_obs.get("w") or 1.0) * float(_right_obs.get("h") or 1.0)))
                # Sparse/high-speed eight-frame input may move the normalized
                # head centre slightly more between two observations.  Allow
                # that extra spatial residual only with stronger appearance
                # agreement, then join the two observations into one person.
                _ordinary_sparse_pair = (
                    _rf - _lf <= 2 and _uv <= 0.18 and
                    _sim is not None and _sim >= 0.84)
                # General multi-frame trajectory prediction. Accumulate the
                # vehicle/global shifts across every missed frame and compare
                # the reappearing head with that predicted image position.
                # Uncertainty grows smoothly with the gap, while the required
                # appearance agreement also becomes stricter. Thus sequence
                # length is irrelevant and no special vehicle/frame is used.
                _gap = int(_rf - _lf)
                _motion_samples = [
                    t.frame_motion_shifts.get(frame)
                    for frame in range(int(_lf) + 1, int(_rf) + 1)
                ]
                _motion_complete = bool(_motion_samples) and all(
                    sample is not None and
                    np.isfinite(float(sample[0])) and
                    np.isfinite(float(sample[1]))
                    for sample in _motion_samples)
                _prediction_residual_norm = float("inf")
                if _motion_complete:
                    _pred_dx = sum(float(sample[0]) for sample in _motion_samples)
                    _pred_dy = sum(float(sample[1]) for sample in _motion_samples)
                    _actual_dx = float(_right_obs.get("cx", 0.0)) - float(
                        _left_obs.get("cx", 0.0))
                    _actual_dy = float(_right_obs.get("cy", 0.0)) - float(
                        _left_obs.get("cy", 0.0))
                    _pair_scale = max(
                        1.0,
                        float(_left_obs.get("w") or 1.0),
                        float(_left_obs.get("h") or 1.0),
                        float(_right_obs.get("w") or 1.0),
                        float(_right_obs.get("h") or 1.0))
                    _prediction_residual_norm = float(np.hypot(
                        _actual_dx - _pred_dx,
                        _actual_dy - _pred_dy)) / _pair_scale
                _gap_uncertainty = float(np.sqrt(max(1, _gap)))
                _required_app = min(0.96, 0.815 + 0.02 * _gap)
                _allowed_prediction_residual = 0.95 + 0.28 * _gap_uncertainty
                _allowed_uv = min(0.34, 0.16 + 0.055 * _gap_uncertainty)
                _allowed_size_ratio = min(1.65, 1.30 + 0.05 * _gap_uncertainty)
                _verified_predicted_pair = (
                    _gap >= 2 and _motion_complete and
                    _prediction_residual_norm <= _allowed_prediction_residual and
                    _uv <= _allowed_uv and
                    _sim is not None and _sim >= _required_app and
                    _pair_min_conf >= 0.40 and
                    _pair_min_person >= 0.50 and
                    _pair_size_ratio <= _allowed_size_ratio and
                    not t._cannot_merge(_left_id, _right_id))
                if _ordinary_sparse_pair or _verified_predicted_pair:
                    promoted.update((_left_id, _right_id))
                    forced_singleton_links.append((_left_id, _right_id))
                    if GENERAL_COUNT_TRACE:
                        print(f"[两帧稀疏碎轨重连] W{t.window_id} "
                              f"ID{_left_id}->ID{_right_id} "
                              f"gap={_rf - _lf} uv={_uv:.3f} sim={_sim:.3f} "
                              f"conf={_pair_min_conf:.3f} person={_pair_min_person:.3f} "
                              f"size={_pair_size_ratio:.2f} "
                              f"pred_res={_prediction_residual_norm:.2f}/"
                              f"{_allowed_prediction_residual:.2f} "
                              f"app_req={_required_app:.3f} "
                              f"predicted={_verified_predicted_pair}")

        # Two detections can exchange tracker ownership in the next frame:
        # the continuing numeric ID is taken by a newcomer, while a fresh
        # singleton ID follows the previous occupant.  Position prediction
        # alone favours the wrong assignment in a fast-moving vehicle, so use
        # the crossed appearance assignment to preserve the extra person.
        verified_cross_swap_singletons = set()
        for old_id in list(stable):
            old_sm = summaries[old_id]
            old_frames = sorted(int(f) for f in old_sm.get("frames", ()))
            old_apps = old_sm.get("appearances", {})
            if len(old_frames) < 2:
                continue
            for f0, f1 in zip(old_frames, old_frames[1:]):
                if f1 != f0 + 1:
                    continue
                direct_sim = self._cosine(old_apps.get(f0), old_apps.get(f1))
                if direct_sim is None or direct_sim >= 0.68:
                    continue
                for new_id, new_sm in summaries.items():
                    new_id = int(new_id)
                    if new_id == old_id or new_id in rejected_track_ids:
                        continue
                    new_frames = sorted(int(f) for f in new_sm.get("frames", ()))
                    if new_frames != [f1]:
                        continue
                    new_obs = new_sm.get("observations", {}).get(f1, {})
                    cross_sim = self._cosine(
                        old_apps.get(f0), new_sm.get("appearances", {}).get(f1))
                    if (cross_sim is not None and cross_sim >= 0.68 and
                            cross_sim >= direct_sim + 0.08 and
                            float(new_obs.get("conf") or 0.0) >= 0.20 and
                            (new_obs.get("person_prob") is None or
                             person_threshold is None or
                             float(new_obs["person_prob"]) >= float(person_threshold))):
                        verified_cross_swap_singletons.add(new_id)
                        promoted.add(new_id)
                        forced_separate_pairs.add(frozenset((old_id, new_id)))
                        if GENERAL_COUNT_TRACE:
                            print(f"[crossed identity swap] W{t.window_id} "
                                  f"old=ID{old_id} newcomer=ID{new_id} "
                                  f"F{f0}->{f1} direct={direct_sim:.3f} "
                                  f"cross={cross_sim:.3f}")

        # A verified handoff may be the next view of a track which ended in the
        # immediately preceding frame.  Prefer that temporal predecessor over
        # a same-frame overlapping track: vehicle motion can make a weak false
        # box inherit the old ID while the real head receives a fresh ID.
        duplicate_handoff_aliases = {}

        # A single oversized/shifted duplicate can sit partly over a mature
        # head for one frame.  Centre-distance alone calls it "separated", but
        # its strong 2-D overlap, much weaker detector score and immediate
        # disappearance identify it as a duplicate false box.  Require the
        # neighbour to span both adjacent frames so a real one-frame passenger
        # beside another passenger is not removed casually.
        for singleton_id in list(promoted):
            singleton_sm = summaries.get(singleton_id, {})
            singleton_frames = sorted(int(f) for f in singleton_sm.get("frames", ()))
            if len(singleton_frames) != 1:
                continue
            frame_no = singleton_frames[0]
            so = singleton_sm.get("observations", {}).get(frame_no)
            if so is None:
                continue
            for stable_id in stable:
                stable_obs_all = summaries[stable_id].get("observations", {})
                to = stable_obs_all.get(frame_no)
                if (to is None or frame_no - 1 not in stable_obs_all or
                        frame_no + 1 not in stable_obs_all):
                    continue
                sw, sh = float(so["w"]), float(so["h"])
                tw, th = float(to["w"]), float(to["h"])
                sx1, sx2 = float(so["cx"]) - sw * .5, float(so["cx"]) + sw * .5
                tx1, tx2 = float(to["cx"]) - tw * .5, float(to["cx"]) + tw * .5
                sy1, sy2 = float(so["cy"]) - sh * .5, float(so["cy"]) + sh * .5
                ty1, ty2 = float(to["cy"]) - th * .5, float(to["cy"]) + th * .5
                xov = max(0.0, min(sx2, tx2) - max(sx1, tx1)) / max(1.0, min(sw, tw))
                yov = max(0.0, min(sy2, ty2) - max(sy1, ty1)) / max(1.0, min(sh, th))
                sconf = float(so.get("conf") or 0.0)
                tconf = float(to.get("conf") or 0.0)
                tprob = to.get("person_prob")
                if (xov >= 0.55 and yov >= 0.60 and sconf <= 0.40 and
                        tconf >= 0.70 and sconf <= 0.50 * tconf and
                        (tprob is None or float(tprob) >= 0.80)):
                    promoted.discard(singleton_id)
                    peak_promoted.discard(singleton_id)
                    rejected_track_ids.add(int(singleton_id))
                    self.last_suppressed_track_ids.add(int(singleton_id))
                    if GENERAL_COUNT_TRACE:
                        print(f"[overlapping singleton false box] W{t.window_id} "
                              f"ID{singleton_id} beside=ID{stable_id} F{frame_no} "
                              f"conf={sconf:.3f}/{tconf:.3f} x={xov:.3f} y={yov:.3f}")
                    break
        for singleton_id in list(promoted):
            singleton_sm = summaries.get(singleton_id, {})
            singleton_frames = sorted(int(f) for f in singleton_sm.get("frames", ()))
            if (len(singleton_frames) != 1 or
                    int(singleton_id) not in getattr(t, "verified_handoff_ids", set())):
                continue
            frame_no = singleton_frames[0]
            singleton_obs = singleton_sm.get("observations", {}).get(frame_no)
            if singleton_obs is None:
                continue
            predecessor_candidates = []
            for predecessor_id in stable:
                predecessor_obs_all = summaries[predecessor_id].get("observations", {})
                if frame_no in predecessor_obs_all or (frame_no - 1) not in predecessor_obs_all:
                    continue
                predecessor_obs = predecessor_obs_all[frame_no - 1]
                boundary = self._adaptive_boundary(
                    t, frame_no - 1, frame_no, predecessor_obs, singleton_obs)
                sim = self._cosine(predecessor_obs.get("feat"), singleton_obs.get("feat"))
                if (boundary["prediction_residual"] <= 1.05 and
                        (sim is None or sim >= 0.42)):
                    predecessor_candidates.append((
                        boundary["prediction_residual"],
                        -(sim if sim is not None else -1.0), int(predecessor_id)))
            if predecessor_candidates:
                _, _, predecessor_id = min(predecessor_candidates)
                duplicate_handoff_aliases[int(singleton_id)] = int(predecessor_id)
                promoted.discard(singleton_id)
                peak_promoted.discard(singleton_id)

                # Suppress only the bad terminal observation, not the mature
                # identity behind it.  It must resume after a gap, be weak in
                # both detector and person filter, and overlap the real handoff
                # horizontally.  This keeps earlier genuine observations.
                sw = float(singleton_obs["w"])
                sx1, sx2 = float(singleton_obs["cx"]) - sw * .5, float(singleton_obs["cx"]) + sw * .5
                for same_id in stable:
                    same_obs_all = summaries[same_id].get("observations", {})
                    same_obs = same_obs_all.get(frame_no)
                    earlier = sorted(f for f in same_obs_all if f < frame_no)
                    if same_obs is None or not earlier or frame_no - earlier[-1] < 2:
                        continue
                    tw = float(same_obs["w"])
                    tx1, tx2 = float(same_obs["cx"]) - tw * .5, float(same_obs["cx"]) + tw * .5
                    x_overlap = max(0.0, min(sx2, tx2) - max(sx1, tx1)) / max(1.0, min(sw, tw))
                    same_conf = float(same_obs.get("conf") or 0.0)
                    same_prob = same_obs.get("person_prob")
                    if (x_overlap >= 0.85 and same_conf <= 0.30 and
                            same_prob is not None and float(same_prob) <= 0.10):
                        rejected_observations.add((int(frame_no), int(same_id)))
                        if GENERAL_COUNT_TRACE:
                            print(f"[weak resumed observation rejected] W{t.window_id} "
                                  f"ID{same_id} F{frame_no} conf={same_conf:.3f} "
                                  f"person={float(same_prob):.4f} x={x_overlap:.3f}")
                if GENERAL_COUNT_TRACE:
                    print(f"[temporal handoff singleton] W{t.window_id} "
                          f"ID{singleton_id}->ID{predecessor_id} F{frame_no}")
                continue
            for stable_id in stable:
                stable_obs = summaries[stable_id].get("observations", {}).get(frame_no)
                if stable_obs is None:
                    continue
                sw, sh = float(singleton_obs["w"]), float(singleton_obs["h"])
                tw, th = float(stable_obs["w"]), float(stable_obs["h"])
                center_ratio = float(np.hypot(
                    float(singleton_obs["cx"] - stable_obs["cx"]),
                    float(singleton_obs["cy"] - stable_obs["cy"]))) / max(1.0, sw, tw)
                sx1, sx2 = float(singleton_obs["cx"]) - sw * .5, float(singleton_obs["cx"]) + sw * .5
                tx1, tx2 = float(stable_obs["cx"]) - tw * .5, float(stable_obs["cx"]) + tw * .5
                sy1, sy2 = float(singleton_obs["cy"]) - sh * .5, float(singleton_obs["cy"]) + sh * .5
                ty1, ty2 = float(stable_obs["cy"]) - th * .5, float(stable_obs["cy"]) + th * .5
                x_overlap = max(0.0, min(sx2, tx2) - max(sx1, tx1)) / max(1.0, min(sw, tw))
                y_overlap = max(0.0, min(sy2, ty2) - max(sy1, ty1)) / max(1.0, min(sh, th))
                size_ratio = min(sw * sh, tw * th) / max(1.0, sw * sh, tw * th)
                if (center_ratio <= 0.70 and x_overlap >= 0.85 and
                        y_overlap >= 0.65 and size_ratio >= 0.60):
                    promoted.discard(singleton_id)
                    peak_promoted.discard(singleton_id)
                    duplicate_handoff_aliases[int(singleton_id)] = int(stable_id)
                    if GENERAL_COUNT_TRACE:
                        print(f"[duplicate handoff singleton] W{t.window_id} "
                              f"ID{singleton_id}->ID{stable_id} F{frame_no} "
                              f"center={center_ratio:.3f} x={x_overlap:.3f} "
                              f"y={y_overlap:.3f} size={size_ratio:.3f}")
                    break

        ids = sorted(stable | promoted)
        if GENERAL_COUNT_TRACE:
            for tid in ids:
                sm = summaries[tid]
                obs_txt = ",".join(
                    f"F{f}:u{sm['observations'][f]['u']:.3f}/x{sm['observations'][f]['cx']:.0f}"
                    f"/c{float(sm['observations'][f].get('conf') or 0.0):.3f}"
                    f"/p{float(sm['observations'][f].get('person_prob') or 0.0):.4f}"
                    f"/s{sm['observations'][f]['w']:.0f}x{sm['observations'][f]['h']:.0f}"
                    for f in sorted(sm.get("observations", {})))
                app_items = sorted(sm.get("appearances", {}).items())
                app_txt = ",".join(
                    f"F{app_items[j-1][0]}-{app_items[j][0]}:"
                    f"{self._cosine(app_items[j-1][1], app_items[j][1]):.3f}"
                    for j in range(1, len(app_items))
                    if self._cosine(app_items[j-1][1], app_items[j][1]) is not None)
                print(f"[轨迹摘要] W{t.window_id} ID{tid} {obs_txt} app[{app_txt}]")
        parent = {tid: tid for tid in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for left_id, right_id in forced_singleton_links:
            if left_id in parent and right_id in parent:
                union(left_id, right_id)

        # 检测器可能连续多帧给同一个头输出两个低 IoU 相邻框。它们虽然
        # “同帧共存”，但若短轨的全部观测始终贴着同一成熟轨迹，并且外观
        # 一致，应作为检测别名合并；只出现一帧的邻近乘员不会触发此规则。
        for i, left_id in enumerate(ids):
            left = summaries[left_id]
            lf = set(int(f) for f in left.get("frames", ()))
            for right_id in ids[i + 1:]:
                right = summaries[right_id]
                rf = set(int(f) for f in right.get("frames", ()))
                shared = sorted(lf & rf)
                if len(shared) < GENERAL_ALIAS_MIN_FRAMES:
                    continue
                shorter = min(len(lf), len(rf))
                if shorter > GENERAL_ALIAS_MAX_FRAMES:
                    continue
                if len(shared) != shorter:
                    continue
                close_all = True
                sims = []
                shared_ious = []
                shared_conf_ratios = []
                for f in shared:
                    lo = left.get("observations", {}).get(f)
                    ro = right.get("observations", {}).get(f)
                    if lo is None or ro is None:
                        close_all = False
                        break
                    dist = float(np.hypot(lo["cx"] - ro["cx"], lo["cy"] - ro["cy"]))
                    if dist > GENERAL_ALIAS_CENTER * max(float(lo["w"]), float(ro["w"])):
                        close_all = False
                    sim = self._cosine(
                        left.get("appearances", {}).get(f),
                        right.get("appearances", {}).get(f))
                    if sim is not None:
                        sims.append(sim)
                    lw, lh = float(lo["w"]), float(lo["h"])
                    rw, rh = float(ro["w"]), float(ro["h"])
                    ix = max(0.0, min(lo["cx"] + lw / 2.0, ro["cx"] + rw / 2.0) -
                             max(lo["cx"] - lw / 2.0, ro["cx"] - rw / 2.0))
                    iy = max(0.0, min(lo["cy"] + lh / 2.0, ro["cy"] + rh / 2.0) -
                             max(lo["cy"] - lh / 2.0, ro["cy"] - rh / 2.0))
                    inter = ix * iy
                    shared_ious.append(inter / max(1.0, lw * lh + rw * rh - inter))
                    lc, rc = lo.get("conf"), ro.get("conf")
                    if lc is not None and rc is not None:
                        shared_conf_ratios.append(
                            min(float(lc), float(rc)) / max(1e-6, float(lc), float(rc)))
                appearance_alias = (
                    close_all and sims and shared_ious and
                    float(np.mean(sims)) >= max(GENERAL_ALIAS_APP, 0.80) and
                    float(np.mean(shared_ious)) >= 0.25)
                # A detector can add a second, similarly-confident box beside a
                # mature head for exactly two frames.  Perspective jitter may
                # move its centre just outside the strict close_all gate even
                # though the boxes keep overlapping and their appearances agree.
                # Treat only a 4+ frame mature track with a fully synchronous
                # two-frame companion as an alias; ordinary short coexisting
                # occupants remain separate.
                persistent_overlap_alias = (
                    shorter == 2 and max(len(lf), len(rf)) >= 4 and
                    len(shared) == shorter and sims and shared_ious and
                    float(np.mean(sims)) >= 0.80 and
                    float(np.mean(shared_ious)) >= 0.28)
                # A false reflection/seat edge often forms a complete two-frame
                # shadow track beside the real head.  Unlike two real occupants,
                # its box substantially overlaps the real box and remains much
                # less confident in every shared frame.
                confidence_shadow = (
                    len(shared_ious) == len(shared) and
                    len(shared_conf_ratios) == len(shared) and
                    float(np.mean(shared_ious)) >= max(GENERAL_ALIAS_SHADOW_IOU, 0.30) and
                    float(np.max(shared_conf_ratios)) <= GENERAL_ALIAS_SHADOW_CONF_RATIO)
                weak_persistent_shadow = (
                    shorter == 2 and max(len(lf), len(rf)) >= 4 and
                    len(shared) == shorter and sims and shared_ious and
                    float(np.mean(sims)) >= 0.65 and
                    float(np.mean(shared_ious)) >= 0.20 and
                    len(shared_conf_ratios) == len(shared) and
                    float(np.max(shared_conf_ratios)) <= 0.55)
                if GENERAL_COUNT_TRACE:
                    print(f"[共存别名候选] W{t.window_id} ID{left_id}/ID{right_id} "
                          f"frames={shared} app={float(np.mean(sims)) if sims else -1:.3f} "
                          f"iou={float(np.mean(shared_ious)) if shared_ious else -1:.3f} "
                          f"conf_ratio={float(np.max(shared_conf_ratios)) if shared_conf_ratios else -1:.3f} "
                          f"appearance_alias={appearance_alias} "
                          f"persistent_overlap={persistent_overlap_alias} "
                          f"weak_shadow={weak_persistent_shadow} "
                          f"shadow={confidence_shadow}")
                # Four stable tracks that are all visible in a four-person peak
                # are stronger evidence than the confidence-shadow heuristic.
                # Crowded real heads often overlap substantially and one can be
                # much weaker, as in vehicles 116/121.
                stable_four_person_peak = (
                    max(t.frame_detected_counts, default=0) >= 4 and
                    left_id in stable and right_id in stable)
                if (appearance_alias or persistent_overlap_alias or
                        weak_persistent_shadow or confidence_shadow) \
                        and not stable_four_person_peak:
                    union(left_id, right_id)

        for i, left_id in enumerate(ids):
            left = summaries[left_id]
            lf = sorted(int(f) for f in left.get("frames", ()))
            if not lf:
                continue
            for right_id in ids[i + 1:]:
                right = summaries[right_id]
                rf = sorted(int(f) for f in right.get("frames", ()))
                if not rf or set(lf) & set(rf):
                    continue
                if (frozenset((left_id, right_id)) in t.cannot_merge_pairs or
                        frozenset((left_id, right_id)) in forced_separate_pairs):
                    continue
                old_id, new_id = (left_id, right_id) if lf[-1] < rf[0] else (right_id, left_id)
                old, new = summaries[old_id], summaries[new_id]
                gap = int(new["first_frame"]) - int(old["last_frame"])
                if gap < 1 or gap > 20:
                    continue
                short_stitch_gap = gap <= GENERAL_STITCH_GAP + 1
                du = float(new["first_u"] - old["last_u"])
                dv = float(new["first_v"] - old["last_v"])
                uv = float(np.hypot(du, dv))
                sim = self._cosine(old.get("last_appearance"), new.get("first_appearance"))
                strong_geometry = short_stitch_gap and uv <= GENERAL_STITCH_UV * 0.55
                geometry_and_app = (short_stitch_gap and uv <= GENERAL_STITCH_UV and
                                    sim is not None and sim >= GENERAL_STITCH_APP)
                # 同一乘员在快速车辆中可能只在两个不相邻帧各露出
                # 一次，中间帧被遮挡。两端都是单帧时没有速度可外推，
                # 仅在外观高度一致且窗内相对位置变化较小时合并。
                singleton_pair_high_app = (
                    len(left.get("frames", ())) == 1 and
                    len(right.get("frames", ())) == 1 and
                    short_stitch_gap and gap <= 2 and uv <= 0.22 and
                    sim is not None and sim >= 0.86)
                # 窗框在画面边缘被裁切时，归一化 u 会突然跳变。对相邻帧、
                # 一端仅一帧的短碎轨，用旧轨最近两点的绝对像素速度外推兜底。
                fragment_motion = False
                anchored_gap_motion = False
                slot_anchor_gap_motion = False
                long_anchor_perspective = False
                edge_entry_long_continuation = False
                cross_view_edge_continuation = False
                mature_singleton_close = False
                fast_motion_slot_anchor = False
                singleton_anchor_motion = False
                edge_singleton_anchor_motion = False
                perspective_anchor_motion = False
                old_frames = sorted(int(f) for f in old.get("frames", ()))
                new_frames = sorted(int(f) for f in new.get("frames", ()))
                if (gap == 1 and min(len(old_frames), len(new_frames)) == 1 and
                        sim is not None and sim >= GENERAL_FRAGMENT_APP and
                        len(old_frames) >= 2):
                    o_prev = old.get("observations", {}).get(old_frames[-2])
                    o_last = old.get("observations", {}).get(old_frames[-1])
                    n_first = new.get("observations", {}).get(new_frames[0])
                    if o_prev is not None and o_last is not None and n_first is not None:
                        pred_x = float(o_last["cx"]) + float(o_last["cx"] - o_prev["cx"])
                        pred_y = float(o_last["cy"]) + float(o_last["cy"] - o_prev["cy"])
                        residual = float(np.hypot(n_first["cx"] - pred_x,
                                                  n_first["cy"] - pred_y))
                        scale = max(float(o_last["w"]), float(n_first["w"]), 1.0)
                        fragment_motion = residual <= GENERAL_FRAGMENT_PX * scale
                # 跨一两个漏检帧的成熟轨迹重连：绝对像素匀速外推必须非常
                # 准确，并要求另一稳定轨迹贯穿缺口作为车辆运动参照。
                if (1 <= gap <= GENERAL_STITCH_GAP + 1 and
                        len(old_frames) >= 1 and len(new_frames) >= 1):
                    o_prev = (old.get("observations", {}).get(old_frames[-2])
                              if len(old_frames) >= 2 else None)
                    o_last = old.get("observations", {}).get(old_frames[-1])
                    n_first = new.get("observations", {}).get(new_frames[0])
                    if o_last is not None and n_first is not None:
                        residual = float("inf")
                        if o_prev is not None:
                            pred_x = float(o_last["cx"]) + gap * float(
                                o_last["cx"] - o_prev["cx"])
                            pred_y = float(o_last["cy"]) + gap * float(
                                o_last["cy"] - o_prev["cy"])
                            residual = float(np.hypot(n_first["cx"] - pred_x,
                                                      n_first["cy"] - pred_y))
                        scale = max(float(o_last["w"]), float(n_first["w"]), 1.0)
                        for anchor_id in ids:
                            if anchor_id in (old_id, new_id):
                                continue
                            anchor = summaries[anchor_id]
                            aobs = anchor.get("observations", {})
                            if not all(f in aobs for f in range(old_frames[-1],
                                                                 new_frames[0] + 1)):
                                continue
                            old_rel = float(o_last["u"] - aobs[old_frames[-1]]["u"])
                            new_rel = float(n_first["u"] - aobs[new_frames[0]]["u"])
                            # 用持续轨迹的实际位移补偿车辆加减速。相比仅按目标
                            # 自身旧速度外推，这对车辆突然加速造成的短暂漏检更稳。
                            anchor_residual = float(np.hypot(
                                (n_first["cx"] - aobs[new_frames[0]]["cx"])
                                - (o_last["cx"] - aobs[old_frames[-1]]["cx"]),
                                (n_first["cy"] - aobs[new_frames[0]]["cy"])
                                - (o_last["cy"] - aobs[old_frames[-1]]["cy"])))
                            rel_change = abs(new_rel - old_rel)
                            anchor_dx = float(aobs[new_frames[0]]["cx"] -
                                              aobs[old_frames[-1]]["cx"])
                            anchor_dy = float(aobs[new_frames[0]]["cy"] -
                                              aobs[old_frames[-1]]["cy"])
                            # Under fast vehicle/perspective motion, normalized
                            # seat coordinates can shift sharply. Adjacent
                            # singleton fragments may still be joined when a
                            # persistent passenger anchors coherent motion, both
                            # detections stay on the same side of it, appearance
                            # agrees moderately, and no coexistence veto exists.
                            fast_motion_slot_anchor = (
                                gap == 1 and len(old_frames) == 1 and
                                len(new_frames) == 1 and sim is not None and
                                sim >= 0.68 and
                                float(o_last.get("conf") or 0.0) >= 0.50 and
                                float(n_first.get("conf") or 0.0) >= 0.50 and
                                old_rel * new_rel > 0.0 and
                                rel_change <= 0.35 and
                                float(np.hypot(anchor_dx, anchor_dy)) >= 2.0 * scale and
                                anchor_residual <= 1.0 * scale and
                                frozenset((int(old_id), int(new_id))) not in
                                t.cannot_merge_pairs)
                            if fast_motion_slot_anchor:
                                break
                            if (min(residual, anchor_residual) <= GENERAL_GAP_PX * scale and
                                    rel_change <= GENERAL_GAP_ANCHOR_U):
                                anchored_gap_motion = True
                                break
                            if (min(len(old_frames), len(new_frames)) == 1 and
                                    sim is not None and sim >= GENERAL_SINGLETON_GAP_APP and
                                    anchor_residual <= GENERAL_SINGLETON_GAP_PX * scale and
                                    rel_change <= GENERAL_SINGLETON_GAP_ANCHOR_U):
                                singleton_anchor_motion = True
                                break
                            # 车窗边缘只露出局部头部时，ReID 外观会明显下降。
                            # 仅对“一端单帧、另一端已稳定”且单帧点贴近
                            # 窗口边缘的情形，允许以更严格的锚点运动代替高外观阈值。
                            singleton_u = (float(o_last["u"]) if len(old_frames) == 1
                                           else float(n_first["u"]))
                            if (min(len(old_frames), len(new_frames)) == 1 and
                                    max(len(old_frames), len(new_frames)) >= 2 and
                                    (singleton_u <= 0.05 or singleton_u >= 0.95) and
                                    sim is not None and sim >= 0.60 and
                                    anchor_residual <= 1.50 * scale and
                                    rel_change <= 0.45):
                                edge_singleton_anchor_motion = True
                                break
                            # 首帧车辆尚未完整进入画面时，同一乘员可能只留下
                            # 一个边缘碎轨；漏一帧后随着观察角度变化重新出现，
                            # 窗内归一化位置会发生较大变化。要求一条成熟轨迹
                            # 贯穿两段及新段末尾作为锚点，并且目标始终位于锚点
                            # 同一侧、相对次序不变，才允许用透视变化解释跳变。
                            same_anchor_side = old_rel * new_rel > 0.0
                            anchor_covers_new = all(
                                f in aobs for f in range(old_frames[-1],
                                                         new_frames[-1] + 1))
                            if (len(old_frames) == 1 and old_frames[0] == 1 and
                                    len(new_frames) >= 2 and gap == 2 and
                                    sim is not None and sim >= 0.60 and
                                    same_anchor_side and anchor_covers_new and
                                    abs(new_rel - old_rel) <= 0.28):
                                perspective_anchor_motion = True
                                break
                # Sparse detections can fragment one passenger across a longer
                # gap while a second, continuous track anchors window motion.
                # Direct coexistence remains a hard cannot-merge constraint.
                # A transitive conflict through a shared companion is weaker:
                # permit stitching only when that same anchor spans the gap and
                # anchor-relative slot plus compensated motion agree; appearance
                # is a supporting sanity check, not the primary identity signal.
                if (GENERAL_STITCH_GAP + 2 <= gap <= GENERAL_ANCHOR_STITCH_MAX_GAP
                        and len(old_frames) + len(new_frames) >= 3
                        and min(len(old_frames), len(new_frames)) <= 2
                        and sim is not None and frozenset((int(old_id), int(new_id))) not in t.cannot_merge_pairs):
                    o_last = old.get("observations", {}).get(old_frames[-1])
                    n_first = new.get("observations", {}).get(new_frames[0])
                    if o_last is not None and n_first is not None:
                        scale = max(1.0, min(
                            max(float(o_last["w"]), float(o_last["h"])),
                            max(float(n_first["w"]), float(n_first["h"]))))
                        confidence_ok = min(
                            float(o_last.get("conf") or 0.0),
                            float(n_first.get("conf") or 0.0)) >= 0.30
                        size_ratio = max(
                            float(o_last["w"]) * float(o_last["h"]),
                            float(n_first["w"]) * float(n_first["h"])) / max(
                                1.0,
                                min(float(o_last["w"]) * float(o_last["h"]),
                                    float(n_first["w"]) * float(n_first["h"])))
                        for anchor_id in ids:
                            if anchor_id in (old_id, new_id):
                                continue
                            aobs = summaries[anchor_id].get("observations", {})
                            anchor_span = list(range(old_frames[-1], new_frames[0] + 1))
                            anchor_coverage = sum(f in aobs for f in anchor_span) / max(1, len(anchor_span))
                            if (old_frames[-1] not in aobs or new_frames[0] not in aobs or
                                    anchor_coverage < 0.65):
                                continue
                            if t._cannot_merge(old_id, new_id):
                                old_neighbors = {int(y) for pair in t.cannot_merge_pairs
                                                 if int(old_id) in pair for y in pair if int(y) != int(old_id)}
                                new_neighbors = {int(y) for pair in t.cannot_merge_pairs
                                                 if int(new_id) in pair for y in pair if int(y) != int(new_id)}
                                if int(anchor_id) not in old_neighbors.intersection(new_neighbors):
                                    continue
                            old_anchor = aobs[old_frames[-1]]
                            new_anchor = aobs[new_frames[0]]
                            old_rel = float(o_last["u"] - old_anchor["u"])
                            new_rel = float(n_first["u"] - new_anchor["u"])
                            slot_drift = abs(new_rel - old_rel)
                            same_side = old_rel * new_rel > 0.0
                            anchor_residual = float(np.hypot(
                                (float(n_first["cx"]) - float(o_last["cx"])) -
                                (float(new_anchor["cx"]) - float(old_anchor["cx"])),
                                (float(n_first["cy"]) - float(o_last["cy"])) -
                                (float(new_anchor["cy"]) - float(old_anchor["cy"]))))
                            allowed_slot_drift = min(
                                0.16, 0.06 + 0.01 * float(np.sqrt(gap)))
                            slot_motion_ok = (
                                same_side and slot_drift <= allowed_slot_drift and
                                anchor_residual <= (0.70 + 0.05 * float(np.sqrt(gap))) * scale)
                            # Appearance is only a sanity check here; the
                            # anchor-relative slot and compensated motion are
                            # the primary evidence under changing viewpoints.
                            appearance_ok = sim >= 0.55
                            if (slot_motion_ok and appearance_ok and confidence_ok
                                    and size_ratio <= 1.80):
                                slot_anchor_gap_motion = True
                                break
                # Dense capture can leave a person undetected for many frames
                # while another occupant remains continuously visible.  Join
                # two mature fragments across that gap only when the same
                # anchor spans every intervening frame, both fragments stay on
                # the same side of it, their relative offset changes smoothly,
                # and endpoint appearance is still supportive.
                if (4 <= gap <= 20 and len(old_frames) >= 5 and
                        len(new_frames) >= 5 and sim is not None and sim >= 0.70):
                    o_last = old.get("observations", {}).get(old_frames[-1])
                    n_first = new.get("observations", {}).get(new_frames[0])
                    if o_last is not None and n_first is not None:
                        for anchor_id in ids:
                            if anchor_id in (old_id, new_id):
                                continue
                            aobs = summaries[anchor_id].get("observations", {})
                            if not all(f in aobs for f in range(old_frames[-1],
                                                                 new_frames[0] + 1)):
                                continue
                            old_rel = float(o_last["u"] - aobs[old_frames[-1]]["u"])
                            new_rel = float(n_first["u"] - aobs[new_frames[0]]["u"])
                            size_ratio = max(float(o_last["w"]), float(n_first["w"])) / max(
                                min(float(o_last["w"]), float(n_first["w"])), 1.0)
                            if (old_rel * new_rel > 0.0 and
                                    abs(new_rel - old_rel) <= 0.26 and
                                    size_ratio <= 1.45):
                                long_anchor_perspective = True
                                break
                # At vehicle entry the window is strongly clipped, so normalized
                # u and lightweight appearance are unreliable.  A short mature
                # edge fragment can reappear several dense frames later as the
                # same head.  Require coherent absolute vehicle travel, similar
                # scale, moderate appearance support, and no coexistence proof.
                if (5 <= gap <= 10 and len(old_frames) >= 2 and
                        len(new_frames) >= 3 and sim is not None and sim >= 0.50 and
                        float(old.get("last_u", 1.0)) <= 0.10 and
                        0.25 <= float(new.get("first_u", 0.0)) <= 0.55 and
                        not t._cannot_merge(old_id, new_id)):
                    o_last = old.get("observations", {}).get(old_frames[-1])
                    n_first = new.get("observations", {}).get(new_frames[0])
                    if o_last is not None and n_first is not None:
                        dx_per_frame = ((float(n_first["cx"]) - float(o_last["cx"])) /
                                        max(1, new_frames[0] - old_frames[-1]))
                        scale_ratio = max(float(o_last["w"]), float(n_first["w"])) / max(
                            1.0, min(float(o_last["w"]), float(n_first["w"])))
                        edge_entry_long_continuation = (
                            -60.0 <= dx_per_frame <= -10.0 and scale_ratio <= 1.80)
                # A head first seen for one frame through the windscreen can
                # reappear a few frames later through the front side window.
                # The crop changes too much for a strict ReID threshold, so use
                # directional edge travel plus moderate appearance evidence.
                # This is intentionally based on normalized geometry and track
                # maturity, not a sequence name, frame number, or identity ID.
                cross_view_edge_continuation = (
                    2 <= gap <= 4 and len(old_frames) == 1 and
                    len(new_frames) >= 3 and sim is not None and sim >= 0.55 and
                    float(old.get("last_u", 0.0)) >= 0.75 and
                    0.30 <= float(new.get("first_u", 0.0)) <= 0.65 and
                    du <= -0.20 and abs(dv) <= 0.22)
                # A mature track followed after one missed frame by a single
                # very close observation is usually fragmentation, not a fifth
                # occupant.  Require both compact window geometry and useful
                # appearance support so this does not absorb genuine newcomers.
                mature_singleton_close = (
                    1 <= gap <= 2 and
                    max(len(old_frames), len(new_frames)) >= 3 and
                    min(len(old_frames), len(new_frames)) == 1 and
                    uv <= 0.13 and sim is not None and sim >= 0.68)
                if GENERAL_COUNT_TRACE:
                    print(f"[碎轨候选] W{t.window_id} ID{old_id}->ID{new_id} "
                          f"gap={gap} uv={uv:.3f} app={sim if sim is not None else -1:.3f} "
                          f"fragment={fragment_motion} anchor_gap={anchored_gap_motion} "
                          f"singleton_anchor={singleton_anchor_motion} "
                          f"slot_anchor_gap={slot_anchor_gap_motion} "
                          f"edge_singleton={edge_singleton_anchor_motion} "
                          f"perspective_anchor={perspective_anchor_motion} "
                          f"long_anchor={long_anchor_perspective} "
                          f"edge_long={edge_entry_long_continuation} "
                          f"cross_view={cross_view_edge_continuation} "
                          f"mature_singleton={mature_singleton_close} "
                          f"fast_motion_anchor={fast_motion_slot_anchor} "
                          f"singleton_pair={singleton_pair_high_app} "
                          f"merge={strong_geometry or geometry_and_app or singleton_pair_high_app or fragment_motion or anchored_gap_motion or singleton_anchor_motion or edge_singleton_anchor_motion or perspective_anchor_motion or fast_motion_slot_anchor or long_anchor_perspective}")
                # A singleton explicitly promoted by a simultaneous occupancy
                # peak is evidence for a new person.  Moderate ReID similarity
                # and motion extrapolation must not silently merge it back into
                # an older track; require genuinely strong appearance instead.
                boundary_bridge = any(
                    bridge_id not in (old_id, new_id) and
                    old_frames[-1] in summaries[bridge_id].get("observations", {}) and
                    new_frames[0] in summaries[bridge_id].get("observations", {})
                    for bridge_id in ids)
                promoted_new_separate = (
                    new_id in promoted and len(new_frames) == 1 and
                    boundary_bridge and sim is not None and sim < 0.70 and
                    not mature_singleton_close and
                    not fast_motion_slot_anchor)
                # In a very fast adjacent-frame translation, a one-frame old
                # passenger and a newly appearing 2+ frame passenger can have
                # similar anchor-relative motion even though they are distinct.
                # A third ID that genuinely spans the boundary proves that the
                # vehicle motion is already represented.  When the candidate's
                # normalized jump is large and appearance is weak, do not let
                # anchor motion collapse the newcomer into the vanished person.
                adjacent_singleton_new_separate = (
                    len(old_frames) == 1 and len(new_frames) >= 2 and gap == 1 and
                    boundary_bridge and uv >= 0.25 and
                    sim is not None and sim < 0.65)
                # Even when absolute pixel extrapolation looks coherent, a
                # large normalized seat-slot jump beside a passenger that spans
                # the boundary is evidence that the new singleton is another
                # occupant rather than the old mature track continuing.
                boundary_peer_large_jump_separate = (
                    len(new_frames) == 1 and gap == 1 and boundary_bridge and
                    uv >= 0.35 and sim is not None and sim < 0.70 and
                    not fast_motion_slot_anchor)
                if promoted_new_separate and GENERAL_COUNT_TRACE:
                    print(f"[峰值新人阻止碎轨合并] W{t.window_id} "
                          f"ID{old_id}->ID{new_id} app={sim:.3f}")
                if adjacent_singleton_new_separate and GENERAL_COUNT_TRACE:
                    print(f"[相邻帧新人阻止碎轨合并] W{t.window_id} "
                          f"ID{old_id}->ID{new_id} uv={uv:.3f} app={sim:.3f}")
                if not (promoted_new_separate or adjacent_singleton_new_separate or
                        boundary_peer_large_jump_separate) and (strong_geometry or geometry_and_app or fragment_motion or
                        anchored_gap_motion or slot_anchor_gap_motion or
                        singleton_anchor_motion or edge_singleton_anchor_motion or
                        perspective_anchor_motion or
                        fast_motion_slot_anchor or
                        long_anchor_perspective or
                        edge_entry_long_continuation or
                        cross_view_edge_continuation or
                        mature_singleton_close or
                        singleton_pair_high_app):
                    union(left_id, right_id)

        # A tracker may briefly rename a person and then recover its original
        # numeric ID (A -> B -> A).  Ordinary one-way fragment stitching loses
        # this merge as soon as the suffix A arrives.  Join the middle segment
        # only when both temporal boundaries are adjacent, non-overlapping and
        # independently agree in pane position and appearance.
        sandwich_aliases = []
        reused_prefix_aliases = []
        for outer_id in ids:
            outer_sm = summaries[outer_id]
            outer_frames = sorted(int(f) for f in outer_sm.get("frames", ()))
            outer_obs = outer_sm.get("observations", {})
            outer_apps = outer_sm.get("appearances", {})
            for inner_id in ids:
                if inner_id == outer_id:
                    continue
                inner_sm = summaries[inner_id]
                inner_frames = sorted(int(f) for f in inner_sm.get("frames", ()))
                if not inner_frames:
                    continue
                before = [f for f in outer_frames if f < inner_frames[0]]
                after = [f for f in outer_frames if f > inner_frames[-1]]
                # If the middle segment coexists with a third trajectory for
                # two consecutive observations, it is an independently
                # visible person rather than a temporary rename of A.
                third_party_coexistence = max((
                    len(set(inner_frames) & set(
                        summaries[third_id].get("frames", ())))
                    for third_id in ids
                    if third_id not in (outer_id, inner_id)), default=0)
                reused_prefix_shape = (
                    bool(before) and bool(after) and len(inner_frames) >= 6 and
                    len(after) >= 2 and after[0] == inner_frames[-1] + 1 and
                    not (set(outer_frames) & set(inner_frames)))
                if ((not before or not after or
                     inner_frames[0] != before[-1] + 1 or
                     after[0] != inner_frames[-1] + 1 or
                     set(outer_frames) & set(inner_frames) or
                     third_party_coexistence > 1) and not reused_prefix_shape):
                    continue
                inner_obs = inner_sm.get("observations", {})
                inner_apps = inner_sm.get("appearances", {})
                left_sim = self._cosine(outer_apps.get(before[-1]),
                                        inner_apps.get(inner_frames[0]))
                right_sim = self._cosine(inner_apps.get(inner_frames[-1]),
                                         outer_apps.get(after[0]))
                left_uv = float(np.hypot(
                    float(outer_obs[before[-1]]["u"] - inner_obs[inner_frames[0]]["u"]),
                    float(outer_obs[before[-1]]["v"] - inner_obs[inner_frames[0]]["v"])))
                right_uv = float(np.hypot(
                    float(inner_obs[inner_frames[-1]]["u"] - outer_obs[after[0]]["u"]),
                    float(inner_obs[inner_frames[-1]]["v"] - outer_obs[after[0]]["v"])))
                if (left_sim is not None and right_sim is not None and
                        left_sim >= 0.68 and right_sim >= 0.68 and
                        left_uv <= 0.10 and right_uv <= 0.10):
                    union(outer_id, inner_id)
                    sandwich_aliases.append((int(inner_id), int(outer_id)))
                    if GENERAL_COUNT_TRACE:
                        print(f"[sandwich trajectory alias] W{t.window_id} "
                              f"ID{outer_id}->ID{inner_id}->ID{outer_id} "
                              f"left_app={left_sim:.3f} right_app={right_sim:.3f} "
                              f"left_uv={left_uv:.3f} right_uv={right_uv:.3f}")

                # Split a reused numeric ID into a sparse false prefix and a
                # genuine suffix.  The suffix must immediately continue a long
                # non-overlapping identity; the prefix must consist only of
                # isolated, weak observations and contain strong non-person
                # evidence.  Thus later genuine frames cannot rescue old false
                # positives through a whole-track median score.
                pre_frames = [f for f in before if f < inner_frames[0]]
                if (len(inner_frames) >= 6 and len(after) >= 2 and
                        after[0] == inner_frames[-1] + 1 and
                        not (set(outer_frames) & set(inner_frames))):
                    prefix_runs = []
                    for pf in pre_frames:
                        if not prefix_runs or pf != prefix_runs[-1][-1] + 1:
                            prefix_runs.append([pf])
                        else:
                            prefix_runs[-1].append(pf)
                    suffix_sim = self._cosine(
                        inner_apps.get(inner_frames[-1]), outer_apps.get(after[0]))
                    suffix_uv = float(np.hypot(
                        float(inner_obs[inner_frames[-1]]["u"] - outer_obs[after[0]]["u"]),
                        float(inner_obs[inner_frames[-1]]["v"] - outer_obs[after[0]]["v"])))
                    prefix_confs = [float(outer_obs[f].get("conf") or 0.0)
                                    for f in pre_frames]
                    prefix_probs = [float(outer_obs[f]["person_prob"])
                                    for f in pre_frames
                                    if outer_obs[f].get("person_prob") is not None]
                    sparse_weak_prefix = (
                        bool(pre_frames) and
                        max((len(r) for r in prefix_runs), default=0) <= 1 and
                        float(np.median(prefix_confs)) <= 0.35 and
                        bool(prefix_probs) and min(prefix_probs) <= 0.10)
                    if GENERAL_COUNT_TRACE and sparse_weak_prefix:
                        print(f"[reused ID prefix candidate] W{t.window_id} "
                              f"outer=ID{outer_id} inner=ID{inner_id} "
                              f"pre={pre_frames} app={suffix_sim} uv={suffix_uv:.3f}")
                    if (sparse_weak_prefix and suffix_sim is not None and
                            suffix_sim >= 0.65 and suffix_uv <= 0.12):
                        union(outer_id, inner_id)
                        reused_prefix_aliases.append((int(outer_id), int(inner_id)))
                        rejected_observations.update(
                            (int(f), int(outer_id)) for f in pre_frames)
                        if GENERAL_COUNT_TRACE:
                            print(f"[reused ID false prefix] W{t.window_id} "
                                  f"drop=ID{outer_id}{pre_frames} "
                                  f"suffix=ID{inner_id}->ID{outer_id} "
                                  f"app={suffix_sim:.3f} uv={suffix_uv:.3f}")

        # A detector can draw two slightly shifted boxes around the same head
        # for several consecutive frames.  Those boxes may receive separate
        # stable tracker IDs and therefore survive the ordinary hit filter.
        # Coexistence is normally proof of two people, so override it only for
        # a persistent, physically-overlapping and synchronously moving pair.
        synchronized_aliases = []
        for alias_pos, left_id in enumerate(ids):
            for right_id in ids[alias_pos + 1:]:
                left_sm, right_sm = summaries[left_id], summaries[right_id]
                left_obs = left_sm.get("observations", {})
                right_obs = right_sm.get("observations", {})
                shared = sorted(set(left_obs) & set(right_obs))
                if len(shared) < 2:
                    continue
                # At least one adjacent shared pair is required: coincidental
                # overlap in two distant frames is not a duplicate lifecycle.
                if not any(b == a + 1 for a, b in zip(shared, shared[1:])):
                    continue
                overlap_scores, center_scores, app_scores = [], [], []
                for frame_no in shared:
                    lo, ro = left_obs[frame_no], right_obs[frame_no]
                    lx1, lx2 = float(lo["cx"]) - .5 * float(lo["w"]), float(lo["cx"]) + .5 * float(lo["w"])
                    rx1, rx2 = float(ro["cx"]) - .5 * float(ro["w"]), float(ro["cx"]) + .5 * float(ro["w"])
                    ly1, ly2 = float(lo["cy"]) - .5 * float(lo["h"]), float(lo["cy"]) + .5 * float(lo["h"])
                    ry1, ry2 = float(ro["cy"]) - .5 * float(ro["h"]), float(ro["cy"]) + .5 * float(ro["h"])
                    x_overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1)) / max(1.0, min(float(lo["w"]), float(ro["w"])))
                    y_overlap = max(0.0, min(ly2, ry2) - max(ly1, ry1)) / max(1.0, min(float(lo["h"]), float(ro["h"])))
                    overlap_scores.append(min(x_overlap, y_overlap))
                    center_scores.append(float(np.hypot(float(lo["cx"]) - float(ro["cx"]),
                                                        float(lo["cy"]) - float(ro["cy"]))) /
                                         max(1.0, float(lo["w"]), float(ro["w"])))
                    app_sim = self._cosine(
                        left_sm.get("appearances", {}).get(frame_no),
                        right_sm.get("appearances", {}).get(frame_no))
                    if app_sim is not None:
                        app_scores.append(float(app_sim))
                motion_residuals = []
                for f0, f1 in zip(shared, shared[1:]):
                    if f1 != f0 + 1:
                        continue
                    ldx = float(left_obs[f1]["cx"] - left_obs[f0]["cx"])
                    ldy = float(left_obs[f1]["cy"] - left_obs[f0]["cy"])
                    rdx = float(right_obs[f1]["cx"] - right_obs[f0]["cx"])
                    rdy = float(right_obs[f1]["cy"] - right_obs[f0]["cy"])
                    scale = max(1.0, float(left_obs[f0]["w"]), float(right_obs[f0]["w"]))
                    motion_residuals.append(float(np.hypot(ldx - rdx, ldy - rdy)) / scale)
                shorter = min(len(left_sm.get("frames", ())), len(right_sm.get("frames", ())))
                longer = max(len(left_sm.get("frames", ())), len(right_sm.get("frames", ())))
                lifecycle_asymmetry = shorter / max(1, longer)
                alias_evidence = (
                    motion_residuals and app_scores and
                    float(np.median(overlap_scores)) >= 0.35 and
                    float(np.median(center_scores)) <= 0.70 and
                    float(np.median(motion_residuals)) <= 0.25 and
                    (float(np.median(app_scores)) >= 0.80 or
                     (float(np.median(overlap_scores)) >= 0.45 and
                      float(np.median(center_scores)) <= 0.55 and
                      float(np.median(app_scores)) >= 0.75)) and
                    lifecycle_asymmetry <= 0.60)
                if GENERAL_COUNT_TRACE and len(shared) >= 2:
                    print(f"[synchronized alias candidate] W{t.window_id} "
                          f"IDs={left_id},{right_id} shared={shared} "
                          f"overlap={float(np.median(overlap_scores)):.3f} "
                          f"center={float(np.median(center_scores)):.3f} "
                          f"motion={float(np.median(motion_residuals)) if motion_residuals else -1:.3f} "
                          f"app={float(np.median(app_scores)) if app_scores else -1:.3f} "
                          f"life={lifecycle_asymmetry:.3f}")
                if alias_evidence:
                    left_component = [tid for tid in ids if find(tid) == find(left_id)]
                    right_component = [tid for tid in ids if find(tid) == find(right_id)]
                    transitive_conflict = any(
                        frozenset((a, b)) != frozenset((left_id, right_id)) and
                        bool(set(summaries[a].get("frames", ())) &
                             set(summaries[b].get("frames", ())))
                        for a in left_component for b in right_component)
                    if transitive_conflict:
                        if GENERAL_COUNT_TRACE:
                            print(f"[synchronized alias transitive veto] W{t.window_id} "
                                  f"IDs={left_id},{right_id} "
                                  f"left_component={left_component} "
                                  f"right_component={right_component}")
                        continue
                    union(left_id, right_id)
                    keep_id, drop_id = ((left_id, right_id)
                                        if len(left_sm.get("frames", ())) >= len(right_sm.get("frames", ()))
                                        else (right_id, left_id))
                    synchronized_aliases.append((int(drop_id), int(keep_id)))
                    if GENERAL_COUNT_TRACE:
                        print(f"[synchronized duplicate alias] W{t.window_id} "
                              f"ID{drop_id}->ID{keep_id} shared={shared} "
                              f"overlap={float(np.median(overlap_scores)):.3f} "
                              f"center={float(np.median(center_scores)):.3f} "
                              f"motion={float(np.median(motion_residuals)):.3f} "
                              f"app={float(np.median(app_scores)):.3f}")

        # Preserve a mature adjacent handoff after the new track grows beyond
        # its first frame.  A stable peer spanning the boundary supplies a
        # vehicle-motion reference; matching relative offset is stronger than
        # raw pane coordinates when the window box changes size or position.
        anchor_handoff_aliases = []
        for old_id in ids:
            old_frames = sorted(int(f) for f in summaries[old_id].get("frames", ()))
            if len(old_frames) < 3:
                continue
            for new_id in ids:
                if new_id == old_id:
                    continue
                new_frames = sorted(int(f) for f in summaries[new_id].get("frames", ()))
                if (len(new_frames) < 2 or new_frames[0] != old_frames[-1] + 1 or
                        set(old_frames) & set(new_frames)):
                    continue
                f0, f1 = old_frames[-1], new_frames[0]
                oo = summaries[old_id]["observations"][f0]
                no = summaries[new_id]["observations"][f1]
                sim = self._cosine(summaries[old_id].get("appearances", {}).get(f0),
                                   summaries[new_id].get("appearances", {}).get(f1))
                if sim is None or sim < 0.55:
                    continue
                for peer_id in ids:
                    if peer_id in (old_id, new_id):
                        continue
                    po = summaries[peer_id].get("observations", {})
                    if f0 not in po or f1 not in po:
                        continue
                    scale = max(1.0, float(oo["w"]), float(no["w"]))
                    old_rel = float(oo["cx"] - po[f0]["cx"])
                    new_rel = float(no["cx"] - po[f1]["cx"])
                    rel_error = abs(new_rel - old_rel) / scale
                    motion_error = abs(
                        float(no["cx"] - oo["cx"]) -
                        float(po[f1]["cx"] - po[f0]["cx"])) / scale
                    if (rel_error <= 0.35 and motion_error <= 0.35 and
                            (old_rel * new_rel > 0.0 or
                             max(abs(old_rel), abs(new_rel)) <= scale)):
                        union(old_id, new_id)
                        anchor_handoff_aliases.append((int(new_id), int(old_id)))
                        if GENERAL_COUNT_TRACE:
                            print(f"[mature anchor handoff] W{t.window_id} "
                                  f"ID{old_id}->ID{new_id} F{f0}->{f1} "
                                  f"app={sim:.3f} rel={rel_error:.3f} "
                                  f"motion={motion_error:.3f} peer=ID{peer_id}")
                        break

        identities = len({find(tid) for tid in ids}) if ids else 0
        if GENERAL_COUNT_TRACE:
            print(f"[融合连通分量] W{t.window_id} ids={ids} "
                  f"roots={{{', '.join(f'{tid}:{find(tid)}' for tid in ids)}}} "
                  f"identities={identities}")

        # 输出使用每个物理身份组中最早/最小的 ID。tracker 内部 ID 不改，
        # 只规范化标注，避免重连后从 102 跳成 103。
        groups = {}
        for tid in ids:
            groups.setdefault(find(tid), []).append(int(tid))
        canonical = {}
        for members in groups.values():
            for tid in members:
                tid_frames = set(summaries[tid].get("frames", ()))
                # 输出重编号只允许非共存片段；同帧共存的两个内部 ID 即使
                # 在计数层被判作检测别名，也不能在画面上显示成同一个号。
                compatible = [
                    other for other in members
                    if not (tid_frames & set(summaries[other].get("frames", ())))
                    or other == tid
                ]
                canonical[tid] = min(compatible) if compatible else tid
        # Synchronized aliases are the one intentional exception to the
        # coexistence display rule: only the canonical survivor may appear in
        # OSD/CSV and contribute to the final distinct-person count.
        for drop_id, keep_id in synchronized_aliases:
            canonical[int(drop_id)] = int(keep_id)
        for drop_id, keep_id in anchor_handoff_aliases:
            canonical[int(drop_id)] = int(keep_id)
        for drop_id, keep_id in sandwich_aliases:
            canonical[int(drop_id)] = int(keep_id)
        for reused_id, genuine_id in reused_prefix_aliases:
            canonical[int(reused_id)] = int(genuine_id)
            canonical[int(genuine_id)] = int(genuine_id)
        for drop_id, keep_id in duplicate_handoff_aliases.items():
            canonical[int(drop_id)] = int(canonical.get(int(keep_id), int(keep_id)))

        # Dense-sequence ID reuse: an early one/two-frame fragment can be
        # continued by another ID, while the tracker later reuses the original
        # numeric ID for a different occupant.  Merge only the early fragment
        # into the continuous bridge and give the later reused suffix a new
        # display ID.  Requiring a long internal gap plus >=3 coexistence frames
        # prevents this from changing ordinary angle/occlusion reconnects.
        dense_reuse_aliases = []
        dense_reuse_join_correction = 0
        for reused_id in ids:
            reused = summaries[reused_id]
            rf = sorted(int(f) for f in reused.get("frames", ()))
            gaps = [(rf[i + 1] - rf[i], i) for i in range(len(rf) - 1)]
            if not gaps:
                continue
            gap_size, gap_idx = max(gaps)
            prefix_frames = rf[:gap_idx + 1]
            suffix_frames = rf[gap_idx + 1:]
            if (gap_size < 8 or len(prefix_frames) > 2 or
                    len(suffix_frames) < GENERAL_MIN_HITS):
                continue
            old_frame = prefix_frames[-1]
            suffix_start = suffix_frames[0]
            old_obs = reused.get("observations", {}).get(old_frame)
            old_app = reused.get("appearances", {}).get(old_frame)
            if old_obs is None or old_app is None:
                continue
            for bridge_id in ids:
                if bridge_id == reused_id:
                    continue
                bridge = summaries[bridge_id]
                bf = sorted(int(f) for f in bridge.get("frames", ()))
                if (not bf or bf[0] <= old_frame or bf[0] - old_frame > 3 or
                        len(set(bf) & set(suffix_frames)) < 3):
                    continue
                bridge_obs = bridge.get("observations", {}).get(bf[0])
                bridge_app = bridge.get("appearances", {}).get(bf[0])
                if bridge_obs is None or bridge_app is None:
                    continue
                uv = float(np.hypot(float(bridge_obs["u"]) - float(old_obs["u"]),
                                    float(bridge_obs["v"]) - float(old_obs["v"])))
                sim = self._cosine(old_app, bridge_app)
                if uv <= 0.06 and sim is not None and sim >= 0.80:
                    canonical[int(bridge_id)] = int(reused_id)
                    # The reused suffix may itself be the continuation of a
                    # short fragment immediately before it.  This happens in
                    # dense sequences while a head grows from a partial box:
                    # fragment A -> fragment B -> reused numeric ID.  Attach
                    # only when the temporal/position/appearance evidence is
                    # tight; the raw reused ID cannot be unioned because its
                    # early prefix belongs to the bridge occupant.
                    suffix_obs = reused.get("observations", {}).get(suffix_start)
                    suffix_app = reused.get("appearances", {}).get(suffix_start)
                    continuation_id = None
                    continuation_score = None
                    if suffix_obs is not None and suffix_app is not None:
                        for fragment_id in ids:
                            if fragment_id in (reused_id, bridge_id):
                                continue
                            fragment = summaries[fragment_id]
                            ff = sorted(int(f) for f in fragment.get("frames", ()))
                            if not ff:
                                continue
                            handoff_gap = suffix_start - ff[-1]
                            if handoff_gap < 1 or handoff_gap > 4:
                                continue
                            fragment_obs = fragment.get("observations", {}).get(ff[-1])
                            fragment_app = fragment.get("appearances", {}).get(ff[-1])
                            if fragment_obs is None or fragment_app is None:
                                continue
                            handoff_uv = float(np.hypot(
                                float(suffix_obs["u"]) - float(fragment_obs["u"]),
                                float(suffix_obs["v"]) - float(fragment_obs["v"])))
                            handoff_sim = self._cosine(fragment_app, suffix_app)
                            if (handoff_uv <= 0.12 and handoff_sim is not None and
                                    handoff_sim >= 0.75):
                                score = float(handoff_sim) - 0.5 * handoff_uv
                                if continuation_score is None or score > continuation_score:
                                    continuation_id = int(fragment_id)
                                    continuation_score = score
                    dense_reuse_aliases.append(
                        (int(reused_id), int(suffix_start), int(bridge_id),
                         continuation_id))
                    if continuation_id is not None:
                        dense_reuse_join_correction += 1
                    if GENERAL_COUNT_TRACE:
                        print(f"[密集帧ID复用拆分] W{t.window_id} "
                              f"ID{reused_id}(prefix)->ID{bridge_id} "
                              f"suffix=F{suffix_start} uv={uv:.3f} app={sim:.3f}" +
                              (f" suffix->ID{continuation_id}"
                               if continuation_id is not None else ""))
                    break
        self.last_general_id_map = canonical
        self.last_general_rejected_observations = rejected_observations

        # 同 ID 内换人：满员连续帧中某轨外观骤变，而同帧其他持续轨迹
        # 仍保持稳定。其他轨迹作为曝光/红外变化参照，避免误判整帧变化。
        frame_set = {int(f) for tid in ids
                     for f in summaries[tid].get("frames", ())}
        observed_peak = max(
            (sum(f in summaries[tid].get("frames", ()) for tid in ids)
             for f in frame_set), default=0)
        # A tracker slot can change occupants after one coexistence frame: the
        # older ID ends, the newly-created ID continues, but from the next frame
        # onward that new slot actually follows the older person.  Recognise
        # only a strict 2+ frame / 3+ frame handoff with strong appearance and
        # extrapolated-position continuity.  Keep both IDs in the overlap frame,
        # then join only the continuing suffix to the ended identity.
        suffix_handoffs = []
        used_handoff_ids = set()
        for _ended_id in ids:
            _ef = sorted(int(f) for f in summaries[_ended_id].get("frames", ()))
            if len(_ef) < 2:
                continue
            _f = _ef[-1]
            for _cont_id in ids:
                if _cont_id == _ended_id or _cont_id in used_handoff_ids:
                    continue
                _cf = sorted(int(f) for f in summaries[_cont_id].get("frames", ()))
                if len(_cf) < 3 or _cf[0] != _f or _f + 1 not in _cf:
                    continue
                _ea = summaries[_ended_id].get("appearances", {})
                _ca = summaries[_cont_id].get("appearances", {})
                _cross = self._cosine(_ea.get(_f), _ca.get(_f + 1))
                _eo = summaries[_ended_id].get("observations", {})
                _co = summaries[_cont_id].get("observations", {})
                if (_f not in _eo or _ef[-2] not in _eo or _f + 1 not in _co):
                    continue
                _pred_u = 2.0 * float(_eo[_f]["u"]) - float(_eo[_ef[-2]]["u"])
                _cross_u = abs(float(_co[_f + 1]["u"]) - _pred_u)
                if (_cross is not None and _cross >= 0.75 and _cross_u <= 0.10):
                    suffix_handoffs.append((int(_cont_id), int(_f + 1),
                                            int(_ended_id)))
                    used_handoff_ids.update((_ended_id, _cont_id))
                    if GENERAL_COUNT_TRACE:
                        print(f"[共存后槽位换人] W{t.window_id} "
                              f"ID{_ended_id}<-ID{_cont_id}@F{_f + 1} "
                              f"app={_cross:.3f} u={_cross_u:.3f}")
                    break
        identity_break_extra = 0
        reuse_splits = []
        three_way_swaps = []
        _used_swap_ids = set()
        for _cont_id in ids:
            _cf = sorted(int(f) for f in summaries[_cont_id].get("frames", ()))
            _ca = summaries[_cont_id].get("appearances", {})
            if len(_cf) < 3:
                continue
            for _f0, _f1 in zip(_cf, _cf[1:]):
                if _f1 != _f0 + 1:
                    continue
                for _ended_id in ids:
                    if _ended_id == _cont_id:
                        continue
                    _ef = sorted(int(f) for f in summaries[_ended_id].get("frames", ()))
                    if len(_ef) < 2 or _ef[-1] != _f0:
                        continue
                    for _new_id in ids:
                        if (_new_id in (_cont_id, _ended_id) or
                                _new_id in _used_swap_ids):
                            continue
                        _nf = sorted(int(f) for f in summaries[_new_id].get("frames", ()))
                        if _nf != [_f1]:
                            continue
                        _ea = summaries[_ended_id].get("appearances", {})
                        _na = summaries[_new_id].get("appearances", {})
                        _straight = self._cosine(_ca.get(_f0), _ca.get(_f1))
                        _cont_new = self._cosine(_ca.get(_f0), _na.get(_f1))
                        _end_cont = self._cosine(_ea.get(_f0), _ca.get(_f1))
                        _end_new = self._cosine(_ea.get(_f0), _na.get(_f1))
                        if None in (_straight, _cont_new, _end_cont, _end_new):
                            continue
                        _direct_score = float(_straight + _end_new)
                        _cross_score = float(_cont_new + _end_cont)
                        _co = summaries[_cont_id].get("observations", {})
                        _eo = summaries[_ended_id].get("observations", {})
                        _no = summaries[_new_id].get("observations", {})
                        _direct_geo = (
                            abs(float(_co[_f1]["u"]) - float(_co[_f0]["u"])) +
                            abs(float(_no[_f1]["u"]) - float(_eo[_f0]["u"])))
                        _cross_geo = (
                            abs(float(_no[_f1]["u"]) - float(_co[_f0]["u"])) +
                            abs(float(_co[_f1]["u"]) - float(_eo[_f0]["u"])))
                        if (_cont_new >= 0.75 and _end_cont >= 0.75 and
                                _cross_score >= _direct_score + 0.08 and
                                _cross_geo + 0.10 <= _direct_geo):
                            three_way_swaps.append((int(_cont_id), int(_f1),
                                                    int(_ended_id), int(_new_id)))
                            _used_swap_ids.update((_cont_id, _ended_id, _new_id))
                            if GENERAL_COUNT_TRACE:
                                print(f"[三轨交叉换位] W{t.window_id} F{_f0}->{_f1} "
                                      f"end={_ended_id} cont={_cont_id} new={_new_id} "
                                      f"direct={_direct_score:.3f} "
                                      f"cross={_cross_score:.3f}")
                            break
        for tid in ids:
            apps = summaries[tid].get("appearances", {})
            frames = sorted(int(f) for f in apps)
            obs = summaries[tid].get("observations", {})
            reused_this_tid = False
            # 两帧短轨在第二帧遇到新的共存人员时，tracker 可能
            # 把新人塞进旧 ID，另一新人则建新 ID。只在旧轨两帧都是
            # 高置信度、外观明显变化，且第二帧确有一个空间分离的
            # 新单帧轨共存时，才把旧 ID 的第二帧拆为新人。
            if len(frames) == 2 and frames[1] == frames[0] + 1:
                f0, f1 = frames
                self_sim = self._cosine(apps.get(f0), apps.get(f1))
                c0 = obs.get(f0, {}).get("conf")
                c1 = obs.get(f1, {}).get("conf")
                has_new_peer = any(
                    pid != tid and
                    sorted(int(f) for f in summaries[pid].get("frames", ())) == [f1]
                    for pid in promoted)
                n0 = sum(f0 in summaries[x].get("frames", ()) for x in ids)
                n1 = sum(f1 in summaries[x].get("frames", ()) for x in ids)
                if (has_new_peer and n0 == 1 and n1 >= 2 and
                        c0 is not None and c1 is not None and
                        min(float(c0), float(c1)) >= 0.75 and
                        self_sim is not None and self_sim < 0.70):
                    identity_break_extra += 1
                    reuse_splits.append((int(tid), int(f1)))
                    reused_this_tid = True
                    if GENERAL_COUNT_TRACE:
                        print(f"[共存交接换人] W{t.window_id} ID{tid} "
                              f"F{f0}->{f1} app={self_sim:.3f}")
                # 满员窗口内，tracker 可能把相邻两帧中占据同一空间槽位的
                # 两个人沿用为同一个 ID。这里不改固定命中帧数，只根据同帧
                # 共存结构补偿人数：
                # 1) 两帧均有至少 3 人，另外至少两条轨迹继续到下一帧，且
                #    目标外观明显比这些参照轨迹更不连续；或
                # 2) 第二帧新出现独立共存轨迹，目标外观也发生明显变化。
                # 两种情况都要求目标只有这相邻两帧，避免拆分长稳定轨迹。
                if not reused_this_tid and self_sim is not None and observed_peak >= 3:
                    slot_u_jump = abs(float(obs[f1]["u"]) - float(obs[f0]["u"]))
                    continuing_peer_sims = []
                    for peer in ids:
                        if peer == tid:
                            continue
                        pframes = sorted(int(f) for f in summaries[peer].get("frames", ()))
                        papps = summaries[peer].get("appearances", {})
                        if f0 in pframes and f1 in pframes and f1 + 1 in pframes:
                            psim = self._cosine(papps.get(f0), papps.get(f1))
                            if psim is not None:
                                continuing_peer_sims.append(psim)
                    peer_median = (float(np.median(continuing_peer_sims))
                                   if continuing_peer_sims else -1.0)
                    full_slot_change = (
                        n0 >= 3 and n1 >= 3 and
                        len(continuing_peer_sims) >= 2 and
                        slot_u_jump <= 0.04 and
                        self_sim < 0.76 and peer_median - self_sim >= 0.045)
                    new_peer_change = (
                        has_new_peer and n0 >= 2 and n1 >= 3 and
                        slot_u_jump <= 0.04 and
                        self_sim < 0.70)
                    if full_slot_change or new_peer_change:
                        identity_break_extra += 1
                        reuse_splits.append((int(tid), int(f1)))
                        reused_this_tid = True
                        if GENERAL_COUNT_TRACE:
                            print(f"[满员槽位换人] W{t.window_id} ID{tid} "
                                  f"F{f0}->{f1} app={self_sim:.3f} "
                                  f"peer={peer_median:.3f} u={slot_u_jump:.3f} "
                                  f"new_peer={has_new_peer}")
                # Two adjacent observations can retain one tracker ID even
                # though the seat occupant changes.  A strong signature is a
                # reciprocal confidence swap against a continuous peer, then a
                # new singleton in the following frame resembling the old slot.
                # This supplements appearance, which can stay deceptively high
                # for neighbouring heads under the same infrared illumination.
                if not reused_this_tid and c0 is not None and c1 is not None:
                    reciprocal_swap = False
                    for peer in ids:
                        if peer == tid:
                            continue
                        pframes = sorted(int(f) for f in summaries[peer].get("frames", ()))
                        pobs = summaries[peer].get("observations", {})
                        if not (f0 in pobs and f1 in pobs and f1 + 1 in pframes):
                            continue
                        pc0 = pobs[f0].get("conf")
                        pc1 = pobs[f1].get("conf")
                        if pc0 is None or pc1 is None:
                            continue
                        reciprocal_swap = (
                            float(c0) >= 0.65 and float(c1) <= 0.40 and
                            float(pc0) <= 0.40 and float(pc1) >= 0.70)
                        if reciprocal_swap:
                            break
                    next_singleton = any(
                        pid != tid and
                        sorted(int(f) for f in summaries[pid].get("frames", ())) == [f1 + 1]
                        for pid in promoted)
                    if reciprocal_swap and next_singleton:
                        identity_break_extra += 1
                        reuse_splits.append((int(tid), int(f1)))
                        reused_this_tid = True
                        if GENERAL_COUNT_TRACE:
                            print(f"[交叉置信槽位换人] W{t.window_id} ID{tid} "
                                  f"F{f0}->{f1} conf={float(c0):.3f}->"
                                  f"{float(c1):.3f}")
            if reused_this_tid:
                continue
            # Adjacent-frame ID reuse can occur when a weak partial head exits
            # and another similar-looking head takes its tracker slot.  Require
            # confidence discontinuity plus disagreement with a continuous peer
            # in both pixel motion and relative window position.
            for f0, f1 in zip(frames, frames[1:]):
                if f1 != f0 + 1 or f0 not in obs or f1 not in obs:
                    continue
                c0, c1 = obs[f0].get("conf"), obs[f1].get("conf")
                if c0 is None or c1 is None:
                    continue
                conf_ratio = min(float(c0), float(c1)) / max(
                    1e-6, float(c0), float(c1))
                if (min(float(c0), float(c1)) > GENERAL_REUSE_ADJ_LOW_CONF or
                        conf_ratio > GENERAL_REUSE_ADJ_CONF_RATIO):
                    continue
                motion_residuals = []
                relative_jumps = []
                dx = float(obs[f1]["cx"] - obs[f0]["cx"])
                dy = float(obs[f1]["cy"] - obs[f0]["cy"])
                scale = max(float(obs[f0]["w"]), float(obs[f1]["w"]), 1.0)
                for peer in ids:
                    if peer == tid:
                        continue
                    pobs = summaries[peer].get("observations", {})
                    if f0 not in pobs or f1 not in pobs:
                        continue
                    pdx = float(pobs[f1]["cx"] - pobs[f0]["cx"])
                    pdy = float(pobs[f1]["cy"] - pobs[f0]["cy"])
                    motion_residuals.append(float(np.hypot(dx - pdx, dy - pdy)) / scale)
                    old_rel = float(obs[f0]["u"] - pobs[f0]["u"])
                    new_rel = float(obs[f1]["u"] - pobs[f1]["u"])
                    relative_jumps.append(abs(new_rel - old_rel))
                if (motion_residuals and relative_jumps and
                        float(np.median(motion_residuals)) >= GENERAL_REUSE_ADJ_MOTION and
                        float(np.median(relative_jumps)) >= GENERAL_REUSE_ADJ_REL_JUMP):
                    identity_break_extra += 1
                    reuse_splits.append((int(tid), int(f1)))
                    reused_this_tid = True
                    if GENERAL_COUNT_TRACE:
                        print(f"[同ID相邻帧复用] W{t.window_id} ID{tid} "
                              f"F{f0}->F{f1} conf_ratio={conf_ratio:.3f} "
                              f"motion={float(np.median(motion_residuals)):.3f} "
                              f"rel_jump={float(np.median(relative_jumps)):.3f}")
                    break
            if reused_this_tid:
                continue
            # A tracker ID can be reused after a missed frame.  Split it only
            # when the earlier segment is established and at least two other
            # occupants span the gap and jointly show a large relative-position
            # jump.  This avoids treating normal vehicle motion as a new person.
            for f0, f1 in zip(frames, frames[1:]):
                missing = f1 - f0 - 1
                if missing < 1 or missing > GENERAL_REUSE_GAP_MAX:
                    continue
                earlier = [f for f in frames if f <= f0]
                if len(earlier) < 2 or earlier[-2] != f0 - 1:
                    continue
                if f0 not in obs or f1 not in obs:
                    continue
                direct_sim = self._cosine(apps.get(f0), apps.get(f1))
                direct_slot_jump = float(np.hypot(
                    float(obs[f1]["u"] - obs[f0]["u"]),
                    float(obs[f1]["v"] - obs[f0]["v"])))
                # Independent two-point identity evidence.  Peer tracks are
                # useful but may themselves be occluded.  Across a gap, a
                # sizeable pane-relative jump requires stronger appearance
                # agreement; otherwise the old ID has been reused by a new
                # occupant.  The threshold changes smoothly with the jump and
                # is deliberately capped below the strong-continuity level.
                required_gap_sim = max(
                    0.80, 0.88 - 0.40 * min(0.30, direct_slot_jump))
                boundary_evidence = self._adaptive_boundary(
                    t, f0, f1, obs[f0], obs[f1])
                # Without a continuous peer as an independent reference,
                # moderate appearance changes across a gap are not enough to
                # declare a new person (viewpoint/window motion can cause them).
                direct_gap_break = (
                    direct_sim is not None and
                    direct_sim < min(required_gap_sim, 0.70) and
                    direct_slot_jump >= 0.12 and
                    boundary_evidence["prediction_residual"] >=
                    (0.85 if boundary_evidence["sparse"] else 0.60) and
                    (boundary_evidence["geometry_reliable"] or
                     boundary_evidence["prediction_residual"] >= 1.20))
                if (GENERAL_COUNT_TRACE and direct_sim is not None and
                        direct_sim < required_gap_sim and direct_slot_jump >= 0.12 and
                        not direct_gap_break):
                    print(f"[adaptive gap veto] W{t.window_id} ID{tid} "
                          f"F{f0}->{f1} slot={direct_slot_jump:.3f} "
                          f"app={direct_sim:.3f}/{required_gap_sim:.3f} "
                          f"motion={boundary_evidence['motion_ratio']:.3f} "
                          f"pred_res={boundary_evidence['prediction_residual']:.3f} "
                          f"geom={boundary_evidence['geometry_reliable']}")
                relative_jumps = []
                for peer in ids:
                    if peer == tid:
                        continue
                    pobs = summaries[peer].get("observations", {})
                    if f0 in pobs and f1 in pobs:
                        old_rel = float(obs[f0]["u"] - pobs[f0]["u"])
                        new_rel = float(obs[f1]["u"] - pobs[f1]["u"])
                        relative_jumps.append(abs(new_rel - old_rel))
                peer_gap_break = (
                    len(relative_jumps) >= 2 and
                    float(np.median(relative_jumps)) >= GENERAL_REUSE_REL_JUMP)
                if direct_gap_break or peer_gap_break:
                    identity_break_extra += 1
                    reuse_splits.append((int(tid), int(f1)))
                    reused_this_tid = True
                    if GENERAL_COUNT_TRACE:
                        print(f"[同ID跨缺口复用] W{t.window_id} ID{tid} "
                              f"F{f0}->F{f1} peers={len(relative_jumps)} "
                              f"rel_jump={float(np.median(relative_jumps)) if relative_jumps else -1:.3f} "
                              f"slot={direct_slot_jump:.3f} "
                              f"app={direct_sim if direct_sim is not None else -1:.3f}/"
                              f"{required_gap_sim:.3f} "
                              f"motion={boundary_evidence['motion_ratio']:.3f} "
                              f"pred_res={boundary_evidence['prediction_residual']:.3f} "
                              f"geom={boundary_evidence['geometry_reliable']}")
                    break
            if reused_this_tid:
                continue
            # 长期稳定轨迹在车辆出画边缘的最后一帧常因裁切产生外观突变，
            # 不据此拆人；同-ID换人补偿只处理短时间交接轨迹。
            if len(frames) > GENERAL_BREAK_MAX_TRACK:
                continue
            for f0, f1 in zip(frames, frames[1:]):
                if f1 != f0 + 1:
                    continue
                sim = self._cosine(apps.get(f0), apps.get(f1))
                if sim is None or sim >= GENERAL_BREAK_APP:
                    continue
                n0 = sum(f0 in summaries[x].get("frames", ()) for x in ids)
                n1 = sum(f1 in summaries[x].get("frames", ()) for x in ids)
                if observed_peak < 2 or n0 != observed_peak or n1 != observed_peak:
                    continue
                peer_sims = []
                peer_motions = []
                for peer in ids:
                    if peer == tid:
                        continue
                    pa = summaries[peer].get("appearances", {})
                    ps = self._cosine(pa.get(f0), pa.get(f1))
                    if ps is not None:
                        peer_sims.append(ps)
                    pobs = summaries[peer].get("observations", {})
                    if f0 in pobs and f1 in pobs:
                        peer_motions.append((
                            float(pobs[f1]["cx"] - pobs[f0]["cx"]),
                            float(pobs[f1]["cy"] - pobs[f0]["cy"])))
                # 同窗乘员应共享车辆的整体位移。目标轨迹若与其他乘员运动
                # 一致，即使外观因姿态、红外或裁切突变，也仍判为同一人。
                obs = summaries[tid].get("observations", {})
                motion_coherent = False
                motion_residual = float("inf")
                if f0 in obs and f1 in obs and peer_motions:
                    dx = float(obs[f1]["cx"] - obs[f0]["cx"])
                    dy = float(obs[f1]["cy"] - obs[f0]["cy"])
                    peer_dx = float(np.median([m[0] for m in peer_motions]))
                    peer_dy = float(np.median([m[1] for m in peer_motions]))
                    scale = max(float(obs[f0]["w"]), float(obs[f1]["w"]), 1.0)
                    motion_residual = float(np.hypot(dx - peer_dx, dy - peer_dy)) / scale
                    motion_coherent = motion_residual <= GENERAL_BREAK_MOTION_RESIDUAL
                # A two-frame slot can be occupied by two different people in
                # consecutive fully-populated frames.  Shared vehicle motion is
                # then misleading.  Split only when at least two peer tracks are
                # continuous and visibly more stable than the target appearance.
                peer_median = (float(np.median(peer_sims))
                               if len(peer_sims) >= 2 else -1.0)
                short_full_peak_break = (
                    len(frames) == 2 and observed_peak >= 3 and
                    len(peer_sims) >= 2 and sim < 0.63 and
                    peer_median >= 0.68 and peer_median - sim >= 0.07)
                if short_full_peak_break:
                    identity_break_extra += 1
                    reuse_splits.append((int(tid), int(f1)))
                    reused_this_tid = True
                    if GENERAL_COUNT_TRACE:
                        print(f"[满员短轨换人] W{t.window_id} ID{tid} "
                              f"F{f0}->F{f1} app={sim:.3f} "
                              f"peer={peer_median:.3f}")
                    break
                if motion_coherent:
                    if GENERAL_COUNT_TRACE:
                        print(f"[同ID外观波动-运动一致] W{t.window_id} ID{tid} "
                              f"F{f0}->{f1} app={sim:.3f} motion={motion_residual:.3f}")
                    continue
                if peer_sims and float(np.median(peer_sims)) >= GENERAL_BREAK_PEER_APP:
                    identity_break_extra += 1
                    if GENERAL_COUNT_TRACE:
                        print(f"[同ID身份断裂] W{t.window_id} ID{tid} "
                              f"F{f0}->{f1} app={sim:.3f} "
                              f"peer={float(np.median(peer_sims)):.3f}")
                    break

        # A mature ID can be handed to another occupant at nearly the same
        # pane-relative slot after the other visible heads have just left the
        # window.  Small u motion alone must not force continuity: when the
        # vehicle-compensated prediction misses, appearance also changes, and
        # the new observation is independently strong, split the ID suffix.
        # This is intentionally limited to a previously crowded window and a
        # recent peer departure, so ordinary single-person perspective changes
        # do not create extra people.
        slot_takeover = None
        if identity_break_extra == 0 and observed_peak >= 3:
            for tid in ids:
                apps = summaries[tid].get("appearances", {})
                obs = summaries[tid].get("observations", {})
                frames = sorted(int(f) for f in apps)
                for pos, (f0, f1) in enumerate(zip(frames, frames[1:])):
                    if (f1 != f0 + 1 or pos + 1 < 6 or
                            f0 not in obs or f1 not in obs):
                        continue
                    sim = self._cosine(apps.get(f0), apps.get(f1))
                    du = abs(float(obs[f1]["u"]) - float(obs[f0]["u"]))
                    new_conf = float(obs[f1].get("conf") or 0.0)
                    new_prob = obs[f1].get("person_prob")
                    boundary = self._adaptive_boundary(t, f0, f1, obs[f0], obs[f1])
                    recent_departed_peer = any(
                        peer != tid and f1 not in summaries[peer].get("observations", {}) and
                        len(summaries[peer].get("frames", ())) >= GENERAL_MIN_HITS and
                        any(max(1, f1 - 5) <= int(pf) <= f0
                            for pf in summaries[peer].get("frames", ()))
                        for peer in ids)
                    if (sim is not None and sim < 0.70 and du <= 0.04 and
                            0.22 <= boundary["prediction_residual"] <= 0.65 and
                            boundary["geometry_reliable"] and
                            new_conf >= 0.55 and
                            (new_prob is None or person_threshold is None or
                             float(new_prob) >= float(person_threshold)) and
                            recent_departed_peer):
                        slot_takeover = (int(tid), int(f1), float(sim), float(du),
                                         float(boundary["prediction_residual"]))
                        break
                if slot_takeover is not None:
                    break
        if slot_takeover is not None:
            tid, split_frame, sim, du, pred_residual = slot_takeover
            identity_break_extra += 1
            reuse_splits.append((tid, split_frame))
            if GENERAL_COUNT_TRACE:
                print(f"[slot-preserving identity takeover] W{t.window_id} ID{tid} "
                      f"F{split_frame - 1}->{split_frame} app={sim:.3f} "
                      f"du={du:.3f} pred={pred_residual:.3f}")

        # In a crowded window, a tracker slot may be reused by a newcomer while
        # a second occupant provides a stable motion reference.  Appearance is
        # only moderately discontinuous under the same infrared illumination,
        # so require all three independent signals: a 3+ person occupancy peak,
        # a large normalized slot jump, and a stable peer spanning the boundary.
        high_motion_turnover = None
        if identity_break_extra == 0 and observed_peak == 3:
            for tid in ids:
                apps = summaries[tid].get("appearances", {})
                obs = summaries[tid].get("observations", {})
                frames = sorted(int(f) for f in apps)
                for f0, f1 in zip(frames, frames[1:]):
                    if f1 != f0 + 1 or f0 not in obs or f1 not in obs:
                        continue
                    sim = self._cosine(apps.get(f0), apps.get(f1))
                    du = abs(float(obs[f1]["u"]) - float(obs[f0]["u"]))
                    if GENERAL_COUNT_TRACE and sim is not None and sim < 0.87:
                        dbg_boundary = self._adaptive_boundary(t, f0, f1, obs[f0], obs[f1])
                        print(f"[adjacent identity diagnostic] W{t.window_id} ID{tid} "
                              f"F{f0}->F{f1} app={sim:.3f} du={du:.3f} "
                              f"pred={dbg_boundary['prediction_residual']:.3f} "
                              f"motion={dbg_boundary['motion_ratio']:.3f} "
                              f"geom={dbg_boundary['geometry_reliable']}")
                    if GENERAL_COUNT_TRACE and frames.index(f0) >= 3:
                        history_frame = frames[frames.index(f0) - 3]
                        history_sim = self._cosine(apps.get(history_frame), apps.get(f1))
                        if history_sim is not None and history_sim < 0.82:
                            print(f"[rolling identity diagnostic] W{t.window_id} ID{tid} "
                                  f"F{history_frame}->F{f1} endpoint={history_sim:.3f} "
                                  f"adjacent={sim if sim is not None else -1:.3f} du={du:.3f}")
                    if sim is None or not (0.68 <= sim < 0.75 and du >= 0.10):
                        continue
                    boundary_evidence = self._adaptive_boundary(
                        t, f0, f1, obs[f0], obs[f1])
                    # A pane-relative slot jump is not identity evidence when
                    # fast vehicle motion or pane-box jitter still predicts the
                    # head correctly in image coordinates.
                    if (not boundary_evidence["geometry_reliable"] or
                            boundary_evidence["prediction_residual"] <
                            (0.85 if boundary_evidence["sparse"] else 0.60)):
                        if GENERAL_COUNT_TRACE:
                            print(f"[adaptive turnover veto] W{t.window_id} ID{tid} "
                                  f"F{f0}->{f1} app={sim:.3f} du={du:.3f} "
                                  f"motion={boundary_evidence['motion_ratio']:.3f} "
                                  f"pred_res={boundary_evidence['prediction_residual']:.3f} "
                                  f"geom={boundary_evidence['geometry_reliable']}")
                        continue
                    peer_sims = []
                    for peer in ids:
                        if peer == tid:
                            continue
                        papps = summaries[peer].get("appearances", {})
                        psim = self._cosine(papps.get(f0), papps.get(f1))
                        if psim is not None:
                            peer_sims.append(psim)
                    if peer_sims and max(peer_sims) >= 0.78:
                        high_motion_turnover = (int(tid), int(f1), sim, du,
                                                max(peer_sims))
                        break
                if high_motion_turnover is not None:
                    break
        if high_motion_turnover is not None:
            tid, split_frame, sim, du, peer_sim = high_motion_turnover
            identity_break_extra += 1
            reuse_splits.append((tid, split_frame))
            if GENERAL_COUNT_TRACE:
                print(f"[满员高运动槽位换人] W{t.window_id} ID{tid} "
                      f"F{split_frame - 1}->{split_frame} app={sim:.3f} "
                      f"du={du:.3f} peer={peer_sim:.3f}")

        if peak_promoted and identities >= observed_peak:
            identity_break_extra = 0

        # Three short IDs can form a diagonal relay while two physical heads
        # coexist: A(f,f+1), B(f+1,f+2), C(f+2,f+3).  Relative to one mature
        # anchor, A->B->C follows one motion lane while A(f+1)->B(f+2) follows
        # the other.  This is an ID permutation, not four occupants.
        relay_identity_correction = 0
        relay_frame_id_map = {}
        if observed_peak == 3 and identities == 4:
            mature_anchors = [tid for tid in ids
                              if len(summaries[tid].get("frames", ())) >= 4]
            short_ids = [tid for tid in ids
                         if len(summaries[tid].get("frames", ())) == 2]
            for anchor_id in mature_anchors:
                ao = summaries[anchor_id].get("observations", {})
                for a_id in short_ids:
                    af = sorted(int(f) for f in summaries[a_id].get("frames", ()))
                    for b_id in short_ids:
                        if b_id == a_id:
                            continue
                        bf = sorted(int(f) for f in summaries[b_id].get("frames", ()))
                        for c_id in short_ids:
                            if c_id in (a_id, b_id):
                                continue
                            cf = sorted(int(f) for f in summaries[c_id].get("frames", ()))
                            if not (af == [af[0], af[0] + 1] and
                                    bf == [af[0] + 1, af[0] + 2] and
                                    cf == [af[0] + 2, af[0] + 3] and
                                    all(f in ao for f in range(af[0], af[0] + 3))):
                                continue
                            aobs = summaries[a_id]["observations"]
                            bobs = summaries[b_id]["observations"]
                            cobs = summaries[c_id]["observations"]
                            scale = max(1.0, float(aobs[af[0]]["w"]),
                                        float(bobs[bf[0]]["w"]),
                                        float(cobs[cf[0]]["w"]))
                            lane1 = [
                                float(aobs[af[0]]["cx"] - ao[af[0]]["cx"]),
                                float(bobs[bf[0]]["cx"] - ao[bf[0]]["cx"]),
                                float(cobs[cf[0]]["cx"] - ao[cf[0]]["cx"]),
                            ]
                            lane2 = [
                                float(aobs[af[1]]["cx"] - ao[af[1]]["cx"]),
                                float(bobs[bf[1]]["cx"] - ao[bf[1]]["cx"]),
                            ]
                            lane1_spread = (max(lane1) - min(lane1)) / scale
                            lane2_error = abs(lane2[1] - lane2[0]) / scale
                            separated = (float(np.mean(lane2)) -
                                         float(np.mean(lane1))) >= 0.20 * scale
                            if (lane1_spread <= 0.50 and lane2_error <= 0.50 and separated):
                                relay_identity_correction = 1
                                relay_frame_id_map.update({
                                    (bf[0], int(b_id)): int(a_id),
                                    (cf[0], int(c_id)): int(a_id),
                                    (af[1], int(a_id)): int(b_id),
                                })
                                canonical[int(c_id)] = int(a_id)
                                if GENERAL_COUNT_TRACE:
                                    print(f"[crossed relay reconstruction] W{t.window_id} "
                                          f"lane1={a_id}->{b_id}->{c_id} "
                                          f"lane2={a_id}->{b_id} anchor={anchor_id} "
                                          f"spread={lane1_spread:.3f}/{lane2_error:.3f}")
                                break
                        if relay_identity_correction:
                            break
                    if relay_identity_correction:
                        break
                if relay_identity_correction:
                    break
        identities = max(0, identities - len(suffix_handoffs) - len(three_way_swaps)
                         - dense_reuse_join_correction - relay_identity_correction)
        identities += identity_break_extra
        next_display_id = max((int(x) for x in summaries), default=t.window_id * 100) + 1
        self.last_general_reuse_map = {}
        for tid, split_frame in reuse_splits:
            self.last_general_reuse_map[int(tid)] = (int(split_frame), next_display_id)
            next_display_id += 1
        self.last_general_suffix_map = {
            int(cont_id): (int(start_frame), int(target_id))
            for cont_id, start_frame, target_id in suffix_handoffs
        }
        self.last_general_frame_id_map = relay_frame_id_map
        for reused_id, suffix_start, bridge_id, continuation_id in dense_reuse_aliases:
            suffix_display_id = (canonical.get(int(continuation_id), int(continuation_id))
                                 if continuation_id is not None else next_display_id)
            self.last_general_suffix_map[int(reused_id)] = (
                int(suffix_start), int(suffix_display_id))
            if continuation_id is None:
                next_display_id += 1
        # For a three-track crossing, the continuing raw ID belongs to the
        # ended neighbour from the boundary frame onward, while the newly
        # created ID inherits the continuing ID's former physical identity.
        for cont_id, start_frame, ended_id, new_id in three_way_swaps:
            self.last_general_suffix_map[int(cont_id)] = (
                int(start_frame), int(ended_id))
            self.last_general_id_map[int(new_id)] = int(cont_id)
        self.last_general_debug = {
            "stable": len(stable), "promoted_singletons": len(promoted),
            "identity_break_extra": int(identity_break_extra),
        }
        # 原始检测峰值也可能含双框；按最终身份连通分量重算共存峰值。
        all_frames = set()
        for tid in ids:
            all_frames.update(int(f) for f in summaries[tid].get("frames", ()))
        peak = max((len({find(tid) for tid in ids
                         if f in summaries[tid].get("frames", ())})
                    for f in all_frames), default=0)
        # Transitive aliases must never collapse a frame that demonstrably
        # contains multiple valid heads.  Build the lower bound before the
        # sandwich union, while still honouring the strict synchronized-box
        # aliases that are known duplicate detections.
        sync_parent = {int(tid): int(tid) for tid in ids}
        for drop_id, keep_id in synchronized_aliases:
            sync_parent[int(drop_id)] = int(keep_id)

        def sync_find(tid):
            tid = int(tid)
            seen = set()
            while sync_parent.get(tid, tid) != tid and tid not in seen:
                seen.add(tid)
                tid = sync_parent.get(tid, tid)
            return tid

        effective_observed_peak = max((
            len({sync_find(tid) for tid in ids
                 if f in summaries[tid].get("frames", ())})
            for f in all_frames), default=0)
        if (GENERAL_COUNT_TRACE and
                effective_observed_peak > max(int(peak), int(identities))):
            print(f"[effective occupancy floor] W{t.window_id} "
                  f"count={max(int(peak), int(identities))} -> "
                  f"{effective_observed_peak}")
        return max(int(peak), int(identities), int(effective_observed_peak))

    def count_diagnostics(self):
        t = self._t
        general = int(self.general_count())
        return {
            "raw_distinct": int(t.count_distinct()),
            "peak": int(max(t.frame_detected_counts, default=0)),
            "general": general,
            **getattr(self, "last_general_debug", {}),
        }

    def last_frame_ids(self, dets):
        """反查「本帧每个检测 → track id」。

        id 编码即 tracker 内部的 `wid*100 + n`（W1 里就是 1xx，W2 就是 2xx）。
        移植块的 update() 只回聚合计数、且内部按 u 排过序，原顺序已丢，
        故在块外用 track_summaries 重建：每个 tid 的
        `observations[frame_index]` 存了该帧的 cx/cy，与传入检测中心一一对应。
        """
        fi = self._t.frame_index
        by_center = {}
        for tid, sm in self._t.track_summaries.items():
            o = (sm or {}).get("observations", {}).get(fi)
            if o is not None:
                # Keep raw IDs until the complete sequence is available.  The
                # incremental canonical map is provisional and later evidence
                # can legitimately reverse an early merge.
                by_center[(round(float(o["cx"]), 1), round(float(o["cy"]), 1))] = int(tid)
        out = []
        for d in dets:
            key = (round((d[0] + d[2]) / 2.0, 1), round((d[1] + d[3]) / 2.0, 1))
            out.append(by_center.get(key))
        return out


def _new_window_tracker(wid):
    if TRACKER_IMPL != "motion":
        return _DGWindowTrackerAdapter(wid)
    return MotionCoherentTracker()


# ==========================================================
# 6. WindowedHeadCounter —— 编排层（窗管理 + 逐窗派发）
# ==========================================================
class WindowedHeadCounter:
    """把 head 按车窗分区，每窗一个 MotionCoherentTracker。

    `update()` 返回的标签与传入的 head 检测**一一对应**；
    落在任何窗外的人头被丢弃（标签为 None）。

    全局 gid 编码为 `wid * 1000 + gid`，保证跨窗唯一、且可被下游按整数解析。
    """

    GID_STRIDE = 1000

    @staticmethod
    def _dedupe_head_indices(indices, boxes, confs, feats):
        """补充中心距离+外观 NMS，处理低 IoU 的同头双框。"""
        kept = []
        for idx in sorted(indices, key=lambda i: float(confs[i]), reverse=True):
            x1, y1, x2, y2 = map(float, boxes[idx])
            w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            duplicate = False
            for other_idx in kept:
                ox1, oy1, ox2, oy2 = map(float, boxes[other_idx])
                ow, oh = max(1.0, ox2 - ox1), max(1.0, oy2 - oy1)
                ocx, ocy = (ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5
                center_dist = float(np.hypot(cx - ocx, cy - ocy))
                y_overlap = max(0.0, min(y2, oy2) - max(y1, oy1)) / min(h, oh)
                size_ratio = min(w * h, ow * oh) / max(w * h, ow * oh)
                app_sim = _DGWindowTrackerAdapter._cosine(
                    feats[idx] if idx < len(feats) else None,
                    feats[other_idx] if other_idx < len(feats) else None)
                if (GENERAL_DUP_TRACE and
                        center_dist <= GENERAL_DUP_CENTER * max(w, ow) and
                        y_overlap >= GENERAL_DUP_Y_OVERLAP and
                        size_ratio >= GENERAL_DUP_SIZE_RATIO):
                    print(f"[近框候选] det={idx}/{other_idx} dist={center_dist:.1f} "
                          f"y_overlap={y_overlap:.3f} size={size_ratio:.3f} "
                          f"app={app_sim if app_sim is not None else -1:.3f}")
                if (center_dist <= GENERAL_DUP_CENTER * max(w, ow) and
                        y_overlap >= GENERAL_DUP_Y_OVERLAP and
                        size_ratio >= GENERAL_DUP_SIZE_RATIO and
                        app_sim is not None and app_sim >= GENERAL_DUP_APP):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(idx)
        return sorted(kept)

    def __init__(self, use_whole_car_control=True):
        self.window_tracker = WindowIdTracker()
        self.window_head_trackers = {}          # {wid: MotionCoherentTracker}
        self.prev_car = None
        # 上一帧主车位移，用于下一帧的"预测位置"（车快速驶过时必须靠它才不会跟丢）
        self.prev_car_shift = (0.0, 0.0)
        # True only for the frame where tracking recovers from a distinct
        # background vehicle to the nearer foreground/main vehicle.
        self.main_car_recovered = False
        self.frame_idx = 0
        # Remember windows confidently assigned to another vehicle.  A vehicle
        # box can disappear for one frame while its window detection persists;
        # without this short memory that background pane may be promoted to a
        # new main-car slot and shift W1..Wn for the whole sequence.
        self.background_window_memory = []  # [{box, frame_idx}]
        # 对照：整车一个 tracker（所有窗的 head 合在一起）
        self.use_whole_car_control = use_whole_car_control
        self.whole_car_tracker = MotionCoherentTracker() if use_whole_car_control else None
        # 诊断
        self.last_windows = []
        self.last_shift = (0.0, 0.0)
        self.last_shift_source = "none"
        # HU 新架构用：本帧「检测索引 → 窗号」的映射（窗分派结果，只读不改行为）。
        # 新模块 `_tracker_graph.py` 需要它才能按窗构建 tracklet。
        self.last_head_window = {}
        # 诊断：逐帧记录 (检出人头数, 主车内人头数, 分到窗内人头数)
        self.diag = []

    # ---------- 主车绑定 ----------
    def _pick_main_car(self, car_dets, img_width=None):
        """取与上一帧主车**预测位置**最近的 car 框；首帧取面积最大者。

        预测位置 = 上一帧主车中心 + 上一帧位移（匀速外推）。

        为什么必须外推：原先比较的是"上一帧中心"，而车快速驶过时每帧位移可达
        200~300px，一帧就超过 CAR_SWITCH_MAX_DIST(400)，于是新框被判成背景车、
        永远沿用旧框（且差距只会越来越大，无法恢复）→ 主车框冻结、窗口匹配失据。
        改成与预测位置比较后，正常行驶的残差接近 0，只有真正跳到别的车才会被拒。
        """
        self.main_car_recovered = False
        if not car_dets:
            return self.prev_car
        if self.prev_car is None:
            return max(car_dets, key=box_area)
        pcx, pcy = center(self.prev_car)
        sx, sy = self.prev_car_shift
        pred_cx, pred_cy = pcx + sx, pcy + sy
        best = min(car_dets,
                   key=lambda b: (center(b)[0] - pred_cx) ** 2 + (center(b)[1] - pred_cy) ** 2)
        # 跟踪可能在主车尚未完整入画时黏到背景车。当前景车辆框显著更大、
        # 且车底更靠下（透视上更接近相机）时，允许从背景车恢复到前景主车。
        largest = max(car_dets, key=box_area)
        # Once an established main vehicle reaches the leading image edge, a
        # following vehicle can become the largest remaining detection.  That
        # is a new vehicle, not a recovery of the current one.  Keep the old
        # main-car lifecycle locked so its follower's windows cannot continue
        # as W2/W3.  Genuine foreground recovery remains allowed away from the
        # leading exit edge.
        travel_dx = float(self.prev_car_shift[0])
        prev_at_leading_exit = False
        best_at_leading_exit = False
        follower_is_behind = False
        if img_width is not None and abs(travel_dx) > 2.0:
            prev_cx = center(self.prev_car)[0]
            largest_cx = center(largest)[0]
            if travel_dx > 0.0:
                prev_at_leading_exit = self.prev_car[2] >= float(img_width) - 15.0
                best_at_leading_exit = best[2] >= float(img_width) - 15.0
                follower_is_behind = largest_cx < prev_cx
            else:
                prev_at_leading_exit = self.prev_car[0] <= 15.0
                best_at_leading_exit = best[0] <= 15.0
                follower_is_behind = largest_cx > prev_cx
        # The previous frame may still be tens of pixels from the edge while
        # the correctly predicted continuation reaches the edge in this frame.
        # Use both boxes; otherwise the larger following vehicle is mistaken
        # for foreground recovery and all valid head histories are cleared.
        block_follower_switch = (prev_at_leading_exit or best_at_leading_exit) and follower_is_behind
        if (largest != best
                and iou(largest, best) <= 0.35
                and box_area(largest) >= 1.40 * max(1.0, box_area(best))
                and largest[3] >= best[3] + 25.0
                and not block_follower_switch):
            self.main_car_recovered = True
            return largest
        # 离预测位置太远 → 视为跳到背景车，不切换
        if np.hypot(center(best)[0] - pred_cx, center(best)[1] - pred_cy) > CAR_SWITCH_MAX_DIST:
            return self.prev_car
        return best

    # ---------- 全局 shift（每帧一次）----------
    def _compute_shift(self, car_now, img_w):
        """用车框位移估整车帧间平移；无车框时返回 None（交给 tracker 回退 RANSAC）。"""
        if car_now is None or self.prev_car is None:
            return None
        left_cropped = (car_now[0] <= 15.0)
        right_cropped = (car_now[2] >= img_w - 15.0)
        if left_cropped and not right_cropped:
            return (car_now[2] - self.prev_car[2], car_now[3] - self.prev_car[3])
        if right_cropped and not left_cropped:
            return (car_now[0] - self.prev_car[0], car_now[1] - self.prev_car[1])
        cx_c, cy_c = center(car_now)
        cx_p, cy_p = center(self.prev_car)
        return (cx_c - cx_p, cy_c - cy_p)

    # ---------- 主入口 ----------
    def update(self, head_dets, head_feats, head_confs,
               win_dets, car_dets, img_w, context_car_dets=None,
               head_person_probs=None):
        """处理一帧。

        返回：labels —— 与 head_dets 等长的列表，元素为 gid(int，含窗号编码) 或 None。
        """
        self.frame_idx += 1

        # ① 主车绑定
        main_car = self._pick_main_car(car_dets, img_w)

        # Confidence can fluctuate on a moving vehicle. A low-confidence car
        # box is accepted as a geometric anchor only when it contains detected
        # windows and is consistent with the locked vehicle's predicted motion.
        # This avoids lowering CAR_CONF globally and admitting background cars.
        if main_car is not None and context_car_dets and win_dets:
            def _window_support(car_box):
                return sum(intersection_ratio(w, car_box) >= WINDOW_COVER_MIN
                           for w in win_dets)

            current_support = _window_support(main_car)
            if self.prev_car is None:
                supported_context = [c for c in context_car_dets
                                     if _window_support(c) > current_support]
                if supported_context:
                    main_car = max(supported_context,
                                   key=lambda c: (_window_support(c),
                                                  box_area(c)))
            else:
                pcx, pcy = center(self.prev_car)
                sx, sy = self.prev_car_shift
                pred_cx, pred_cy = pcx + sx, pcy + sy
                current_dist = float(np.hypot(center(main_car)[0] - pred_cx,
                                              center(main_car)[1] - pred_cy))
                supported_context = []
                for candidate in context_car_dets:
                    if iou(candidate, main_car) >= 0.95:
                        continue
                    support = _window_support(candidate)
                    dist = float(np.hypot(center(candidate)[0] - pred_cx,
                                          center(candidate)[1] - pred_cy))
                    if (support >= current_support and support > 0 and
                            dist <= CAR_SWITCH_MAX_DIST and
                            dist + 20.0 < current_dist):
                        supported_context.append((support, -dist, candidate))
                if supported_context:
                    main_car = max(supported_context,
                                   key=lambda item: (item[0], item[1]))[2]
                    if GENERAL_COUNT_TRACE:
                        print(f"[低置信主车框连续性补偿] frame={self.frame_idx} "
                              f"car={tuple(round(float(v), 1) for v in main_car)}")

        # A sequence can start with only a small background vehicle visible.
        # If the locked vehicle covers none of the detected windows, but a
        # geometrically distinct and substantially larger vehicle covers them,
        # recover to that window-supported vehicle.  This uses only car/window
        # geometry and avoids any passenger-count or scene-specific prior.
        if main_car is not None and win_dets and car_dets:
            main_window_support = sum(
                intersection_ratio(w, main_car) >= WINDOW_COVER_MIN
                for w in win_dets
            )
            supported_cars = []
            for candidate in car_dets:
                support = sum(
                    intersection_ratio(w, candidate) >= WINDOW_COVER_MIN
                    for w in win_dets
                )
                if (support > 0 and iou(candidate, main_car) <= 0.35 and
                        box_area(candidate) >= 1.50 * max(1.0, box_area(main_car))):
                    supported_cars.append((support, box_area(candidate), candidate))
            if main_window_support == 0 and supported_cars:
                recovered = max(supported_cars, key=lambda item: (item[0], item[1]))[2]
                # Recovery is only for an early provisional/background lock.
                # Once the current vehicle has produced several independent
                # passenger identities, a later non-overlapping car is normally
                # the following/background vehicle and must not erase the
                # completed main-car lifecycle.
                confirmed_people = (self.whole_car_tracker.total_count()
                                    if self.whole_car_tracker is not None else 0)
                recovered_head_support = sum(
                    intersection_ratio(h, recovered) >= 0.50
                    for h in head_dets)
                allow_provisional_recovery = (
                    self.frame_idx <= 6 or confirmed_people <= 1 or
                    recovered_head_support >= 2)
                # Do not concatenate a following vehicle to an established
                # main car which is already leaving through the leading edge.
                # Early frame number alone is not evidence of a provisional
                # lock in sparse 8-frame sequences.
                travel_dx = float(self.prev_car_shift[0])
                exiting_established_car = False
                follower_behind_exit = False
                if self.prev_car is not None and abs(travel_dx) > 2.0:
                    prev_cx = center(self.prev_car)[0]
                    recovered_cx = center(recovered)[0]
                    if travel_dx > 0.0:
                        exiting_established_car = self.prev_car[2] >= img_w - 15.0
                        follower_behind_exit = recovered_cx < prev_cx
                    else:
                        exiting_established_car = self.prev_car[0] <= 15.0
                        follower_behind_exit = recovered_cx > prev_cx
                if (confirmed_people >= 2 and exiting_established_car and
                        follower_behind_exit):
                    allow_provisional_recovery = False
                if allow_provisional_recovery:
                    if GENERAL_COUNT_TRACE:
                        print(f"[主车车窗归属恢复] frame={self.frame_idx} "
                              f"car={tuple(round(float(v), 1) for v in main_car)} -> "
                              f"{tuple(round(float(v), 1) for v in recovered)}")
                    main_car = recovered
                    self.main_car_recovered = True
                elif GENERAL_COUNT_TRACE:
                    print(f"[后车切换抑制] frame={self.frame_idx} "
                          f"confirmed_people={confirmed_people} "
                          f"candidate_heads={recovered_head_support}")

        # The sequence may begin while only a distant/background vehicle is
        # detected.  Once a geometrically distinct, larger and lower vehicle
        # is selected by _pick_main_car, all window/head identities accumulated
        # on the provisional background car are invalid.  Start the main-car
        # lifecycle cleanly; the fixed 3-frame/2-frame count requirements are
        # deliberately unchanged.
        if self.main_car_recovered:
            if GENERAL_COUNT_TRACE:
                print(f"[主车前景恢复] frame={self.frame_idx} reset provisional "
                      f"car={tuple(round(float(v), 1) for v in self.prev_car)} -> "
                      f"{tuple(round(float(v), 1) for v in main_car)}")
            self.window_tracker = WindowIdTracker()
            self.window_head_trackers.clear()
            if self.use_whole_car_control:
                self.whole_car_tracker = MotionCoherentTracker()
            self.prev_car_shift = (0.0, 0.0)

        # ② 全局 shift（本帧一次，供所有窗共用）
        shift = self._compute_shift(main_car, img_w)
        self.last_shift = shift if shift is not None else (0.0, 0.0)
        self.last_shift_source = "car" if shift is not None else "ransac-fallback"
        # 记录本帧位移，供下一帧主车位置外推（见 _pick_main_car）
        if shift is not None and (abs(float(shift[0])) > 2.0 or
                                  abs(float(shift[1])) > 2.0):
            self.prev_car_shift = shift

        # ③ 窗过滤 + 跨帧稳定编号
        filtered_windows = [
            w for w in win_dets
            if main_car is not None and intersection_ratio(w, main_car) >= WINDOW_COVER_MIN
        ]
        # 主车的大矩形框可能同时覆盖后方车辆。若一个车窗主要落在另一辆
        # 明显更小的车辆框内，则该车窗属于背景车，不能进入主车窗口跟踪。
        # 这里完全按车辆/车窗几何归属判断，与人头置信度无关。
        if main_car is not None and context_car_dets:
            if GENERAL_COUNT_TRACE:
                print(f"[车辆车窗归属] frame={self.frame_idx} main="
                      f"{tuple(round(float(v), 1) for v in main_car)} "
                      f"cars={[tuple(round(float(v), 1) for v in c) for c in context_car_dets]} "
                      f"windows={[tuple(round(float(v), 1) for v in w) for w in filtered_windows]}")
            def _belongs_to_other_car(w, c):
                if c == main_car or iou(c, main_car) > 0.35:
                    return False
                if intersection_ratio(w, c) >= 0.80:
                    return True
                # Vehicle detectors can truncate a truck at the cab/body seam.
                # Its window then starts immediately outside the vehicle box,
                # while retaining almost the same vertical span.  Recognise
                # that tight edge continuation without expanding the whole box
                # (which would swallow an overlapping foreground car window).
                ww = max(1.0, float(w[2] - w[0]))
                cw = max(1.0, float(c[2] - c[0]))
                wh = max(1.0, float(w[3] - w[1]))
                y_overlap = max(0.0, min(w[3], c[3]) - max(w[1], c[1])) / wh
                edge_gap = min(abs(float(w[0] - c[2])),
                               abs(float(c[0] - w[2])))
                window_cx = center(w)[0]
                # A box merely touching the boundary of another vehicle is
                # ambiguous when the two cars overlap in perspective.  Keep
                # the edge-continuation exception only when the window center
                # is also close to that vehicle's boundary; otherwise a main-
                # car pane straddling the seam can be mislabeled as background.
                center_edge_gap = min(abs(window_cx - float(c[0])),
                                      abs(window_cx - float(c[2])))
                return (y_overlap >= 0.80 and edge_gap <= max(8.0, 0.02 * cw)
                        and center_edge_gap <= max(20.0, 0.12 * cw)
                        and ww <= 0.50 * cw)

            # Keep a short-lived trajectory for windows explicitly owned by a
            # different detected car.  If that car box drops out next frame,
            # motion/scale continuity can still keep its window from entering
            # the main vehicle's numbering and occupant trackers.
            current_background = []
            retained_background = [
                item for item in self.background_window_memory
                if 0 < self.frame_idx - int(item['frame_idx']) <= 2
            ]
            kept_windows = []
            for w in filtered_windows:
                other_owner = any(_belongs_to_other_car(w, c)
                                  for c in context_car_dets)
                if other_owner:
                    current_background.append({
                        'box': tuple(float(v) for v in w),
                        'frame_idx': self.frame_idx,
                    })
                    if GENERAL_COUNT_TRACE:
                        print(f"[背景车窗排除] frame={self.frame_idx} "
                              f"box={tuple(round(float(v), 1) for v in w)} "
                              "reason=other_vehicle_box")
                    continue
                matched_memory = None
                for item in retained_background:
                    gap = self.frame_idx - int(item['frame_idx'])
                    old = item['box']
                    old_w = max(1.0, float(old[2] - old[0]))
                    old_h = max(1.0, float(old[3] - old[1]))
                    new_w = max(1.0, float(w[2] - w[0]))
                    new_h = max(1.0, float(w[3] - w[1]))
                    pred_x = center(old)[0] + float(self.last_shift[0]) * gap
                    pred_y = center(old)[1] + float(self.last_shift[1]) * gap
                    residual = float(np.hypot(center(w)[0] - pred_x,
                                              center(w)[1] - pred_y))
                    scale_ok = (min(old_w, new_w) / max(old_w, new_w) >= 0.55
                                and min(old_h, new_h) / max(old_h, new_h) >= 0.55)
                    if (scale_ok and
                            residual <= max(32.0, 0.35 * max(old_w, new_w))):
                        matched_memory = item
                        item['box'] = tuple(float(v) for v in w)
                        item['frame_idx'] = self.frame_idx
                        if GENERAL_COUNT_TRACE:
                            print(f"[背景车窗延续排除] frame={self.frame_idx} "
                                  f"box={tuple(round(float(v), 1) for v in w)} "
                                  f"residual={residual:.1f} gap={gap}")
                        break
                if matched_memory is None:
                    kept_windows.append(w)
            self.background_window_memory = retained_background + current_background
            filtered_windows = kept_windows
        # A pillar/reflection can occasionally be detected as a nearly square,
        # very narrow "window" between two normal side windows.  If admitted,
        # the window tracker preserves that false slot and shifts every later
        # window number.  Reject only a geometrically sandwiched outlier when
        # three or more simultaneous window detections provide both neighbours;
        # this is independent of head detections/confidence.
        if len(filtered_windows) >= 3:
            ordered_windows = sorted(filtered_windows, key=lambda w: center(w)[0])
            pillar_false_windows = set()
            for pos in range(1, len(ordered_windows) - 1):
                left_w, mid_w, right_w = ordered_windows[pos - 1:pos + 2]
                mid_width = max(1.0, float(mid_w[2] - mid_w[0]))
                mid_height = max(1.0, float(mid_w[3] - mid_w[1]))
                neighbour_widths = [max(1.0, float(w[2] - w[0]))
                                    for w in (left_w, right_w)]
                vertical_overlap = min(
                    max(0.0, min(mid_w[3], w[3]) - max(mid_w[1], w[1])) /
                    mid_height for w in (left_w, right_w))
                if (mid_width / mid_height <= 1.35 and
                        mid_width <= 0.55 * min(neighbour_widths) and
                        vertical_overlap >= 0.75):
                    pillar_false_windows.add(tuple(mid_w))
                    if GENERAL_COUNT_TRACE:
                        print(f"[车窗几何误检过滤] frame={self.frame_idx} "
                              f"box={tuple(round(float(v), 1) for v in mid_w)} "
                              f"size={mid_width:.0f}x{mid_height:.0f}")
            if pillar_false_windows:
                filtered_windows = [w for w in filtered_windows
                                    if tuple(w) not in pillar_false_windows]
        # With four or more side-window detections, a small near-square box can
        # also appear just beyond the leading real window (typically windshield
        # reflection/trim).  If it is far narrower than its sole neighbour and
        # is not merely a box clipped by the image boundary, exclude it before
        # assigning W1; otherwise every real window is shifted by one.
        if len(filtered_windows) >= 4:
            ordered_windows = sorted(filtered_windows, key=lambda w: center(w)[0])
            edge_false_windows = set()
            for edge_pos, neighbour_pos in ((0, 1), (-1, -2)):
                edge_w = ordered_windows[edge_pos]
                neighbour_w = ordered_windows[neighbour_pos]
                edge_width = max(1.0, float(edge_w[2] - edge_w[0]))
                edge_height = max(1.0, float(edge_w[3] - edge_w[1]))
                neighbour_width = max(1.0, float(neighbour_w[2] - neighbour_w[0]))
                touches_image_edge = edge_w[0] <= 15.0 or edge_w[2] >= float(img_w) - 15.0
                if (not touches_image_edge and edge_width / edge_height <= 1.45 and
                        edge_width <= 0.45 * neighbour_width):
                    edge_false_windows.add(tuple(edge_w))
                    if GENERAL_COUNT_TRACE:
                        print(f"[车窗边缘几何误检过滤] frame={self.frame_idx} "
                              f"box={tuple(round(float(v), 1) for v in edge_w)} "
                              f"size={edge_width:.0f}x{edge_height:.0f}")
            if edge_false_windows:
                filtered_windows = [w for w in filtered_windows
                                    if tuple(w) not in edge_false_windows]
        is_car_front_on_left = True
        if main_car is not None and filtered_windows:
            is_car_front_on_left = (
                sum(center(w)[0] for w in filtered_windows) / len(filtered_windows)
                >= center(main_car)[0]
            )
        windows_with_ids = self.window_tracker.update(
            filtered_windows, main_car, img_w, is_car_front_on_left=is_car_front_on_left)
        # WindowIdTracker may reject a cropped/jittering vehicle box and replace
        # its horizontal shift with multi-window consensus.  Feed the same
        # corrected motion to all head trackers so window and occupant tracking
        # stay in one coordinate system.
        if (windows_with_ids and
                not self.window_tracker.last_motion_clipped_fallback):
            corrected_dx = float(self.window_tracker.last_motion[0])
            if shift is None:
                shift = (corrected_dx, float(self.window_tracker.last_motion[1]))
            elif abs(corrected_dx - float(shift[0])) > 1e-6:
                shift = (corrected_dx, float(shift[1]))
            self.last_shift = shift
        self.last_windows = windows_with_ids

        # ④ head 分窗：落进哪个窗就归哪个；窗外丢弃
        heads_by_window = defaultdict(list)
        keep_idx_by_window = defaultdict(list)
        for hi, hb in enumerate(head_dets):
            hc = center(hb)
            cands = [(wid, wb) for wid, wb in windows_with_ids if point_in_box(hc, wb)]
            if not cands:
                continue
            best_wid, _ = min(
                cands,
                key=lambda kv: (center(kv[1])[0] - hc[0]) ** 2 + (center(kv[1])[1] - hc[1]) ** 2)
            heads_by_window[best_wid].append(hb)
            keep_idx_by_window[best_wid].append(hi)

        # HU 新架构用：把「检测索引 → 窗号」暴露出来（纯记录，不影响任何既有逻辑）
        self.last_head_window = {hi: wid
                                 for wid, idxs in keep_idx_by_window.items()
                                 for hi in idxs}

        # ④b 诊断：逐帧记 检出 / 主车内 / 分到窗内 三档人头数
        #     用途：把「少算」归因到 检测漏检(该帧就没检出) vs 窗盲区(检出在主车内但没落进任何窗)
        _n_in_car = 0
        if main_car is not None:
            for _hb in head_dets:
                if point_in_box(center(_hb), main_car):
                    _n_in_car += 1
        self.diag.append((len(head_dets), _n_in_car,
                          sum(len(_v) for _v in keep_idx_by_window.values())))

        # ⑤ 逐窗派发（同一个 shift 分发给所有窗）
        labels = [None] * len(head_dets)
        for wid, wb in windows_with_ids:
            tr = self.window_head_trackers.get(wid)
            if tr is None:
                tr = _new_window_tracker(wid)
                self.window_head_trackers[wid] = tr
            idxs = keep_idx_by_window.get(wid, [])
            if GENERAL_PRETRACK_DEDUPE:
                idxs = self._dedupe_head_indices(
                    idxs, head_dets, head_confs, head_feats)
            dets_w = [head_dets[i] for i in idxs]
            feats_w = [head_feats[i] for i in idxs]
            confs_w = [head_confs[i] for i in idxs]
            probs_w = ([head_person_probs[i] for i in idxs]
                       if head_person_probs is not None else None)

            if isinstance(tr, _DGWindowTrackerAdapter):
                # dualgate 版：自己内部做窗归一化，需一并给窗框/方向
                tr.update(dets_w, feats_w, confs_w, person_probs=probs_w,
                          motion_shift=shift,
                          window_box=wb, img_w=img_w,
                          is_head_on_left=is_car_front_on_left)
                # 反查本帧各检测的 track id（wid*100+n），供下游标注/可视化
                for i, tid in zip(idxs, tr.last_frame_ids(dets_w)):
                    if tid is not None:
                        labels[i] = int(tid)
                continue
            if not idxs:
                tr.update([], [], [], motion_shift=shift)
                continue
            gids = tr.update(dets_w, feats_w, confs_w, motion_shift=shift)
            for i, gid in zip(idxs, gids):
                labels[i] = int(wid) * self.GID_STRIDE + int(gid)

        # ⑥ 对照：整车一个 tracker 跑同一批 head
        if self.whole_car_tracker is not None:
            self.whole_car_tracker.update(head_dets, head_feats, head_confs,
                                          motion_shift=shift)

        self.prev_car = main_car
        return labels

    # ---------- 计数 ----------
    def total_per_window(self):
        """正式口径：逐窗通用融合人数之和。"""
        return sum(self._tracker_count(t) for t in self.window_head_trackers.values())

    @staticmethod
    def _tracker_count(tracker):
        if hasattr(tracker, "general_count"):
            return int(tracker.general_count())
        return int(tracker.total_count())

    def total_per_window_raw(self):
        """旧口径：每窗历史 distinct ID 直接求和，供回归对照。"""
        return sum(t.total_count() for t in self.window_head_trackers.values())

    def total_whole_car(self):
        """对照口径：整车一个 ID 空间下的 distinct gid。"""
        if self.whole_car_tracker is None:
            return None
        return self.whole_car_tracker.total_count()

    def per_window_breakdown(self):
        return {wid: self._tracker_count(t)
                for wid, t in sorted(self.window_head_trackers.items())}

    def per_window_diagnostics(self):
        out = {}
        for wid, tracker in sorted(self.window_head_trackers.items()):
            if hasattr(tracker, "count_diagnostics"):
                out[wid] = tracker.count_diagnostics()
            else:
                raw = int(tracker.total_count())
                out[wid] = {"raw_distinct": raw, "peak": None, "general": raw}
        return out

    def diag_stats(self):
        """诊断汇总：全程 检出 / 主车内 / 分到窗内 三档人头数的峰值与总量。

        `peak_in_car` 是"任意单帧里同时在主车框内的人头数"的最大值 ——
        它 ≤ 真值时说明检测端从未同时检出足够的人（检测漏检）；
        `peak_in_car > peak_in_win` 则说明有人在主车内、却没落进任何窗（窗盲区）。
        """
        if not self.diag:
            return {}
        det, inc, inw = zip(*self.diag)
        return {
            "frames": len(self.diag),
            "peak_det": int(max(det)),
            "peak_in_car": int(max(inc)),
            "peak_in_win": int(max(inw)),
            "sum_det": int(sum(det)),
            "sum_in_car": int(sum(inc)),
            "sum_in_win": int(sum(inw)),
        }


# ==========================================================
# 7. 画框 + 标签（沿用原脚本的智能放置）
# ==========================================================
def _rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def draw_boxes_with_labels(img, dets, labels, color=(0, 255, 0), confs=None):
    FONT, SCALE, THICK, BOX_THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1
    for (x1, y1, x2, y2) in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICK)
    placed = []
    for idx, (box, label) in enumerate(zip(dets, labels)):
        # 将 YOLO 头肩检测置信度直接标在对应检测框上，保留原有窗口/轨迹 ID。
        if confs is not None and idx < len(confs):
            conf_text = f"{float(confs[idx]):.3f}"
            label = f"{label} {conf_text}" if label else f"Head {conf_text}"
        if not label:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        (tw, th), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
        candidates = [(x1, y1 - th - 3), (x1, y1 - th - 11), (x1, y2 + 3),
                      (x1, y2 + 11), (x1 + 2, y1 + th + 2),
                      (x1 - tw - 3, y1 + th), (x2 + 3, y1 + th)]
        chosen = candidates[0]
        for (lx, ly) in candidates:
            lrect = (lx, ly, lx + tw, ly + th)
            if any(_rects_overlap(lrect, dets[j]) for j in range(len(dets)) if j != idx):
                continue
            if any(_rects_overlap(lrect, p) for p in placed):
                continue
            chosen = (lx, ly)
            break
        lx, ly = chosen
        ov = img[max(0, ly):ly + th, max(0, lx):lx + tw]
        if ov.size > 0:
            img[max(0, ly):ly + th, max(0, lx):lx + tw] = cv2.addWeighted(
                ov, 0.4, np.zeros_like(ov), 0.6, 0)
        cv2.putText(img, label, (lx, ly + th), FONT, SCALE, color, THICK)
        placed.append((lx, ly, lx + tw, ly + th))


def draw_windows(img, windows_with_ids):
    """把窗框和窗号画出来，便于人工核对分区是否正确。"""
    window_color = (0, 255, 255)  # yellow in OpenCV BGR
    for wid, box in windows_with_ids:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), window_color, 1)
        cv2.putText(img, f"W{wid}", (x1 + 3, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, window_color, 1)


def check_overload(window_people_dict):
    """与 RK3588 版本一致的分排超员口径。"""
    if not window_people_dict:
        return False, "无人员"
    occupied = sorted(int(wid) for wid, count in window_people_dict.items()
                      if int(count) > 0)
    if not occupied:
        return False, "无人"

    reasons = []
    if int(window_people_dict.get(1, 0)) > 2:
        reasons.append(f"前排[W1]({int(window_people_dict[1])}人>2人)")

    last_wid = max(occupied)
    if last_wid > 1:
        last_count = int(window_people_dict.get(last_wid, 0))
        if last_count > 3:
            reasons.append(f"最后一排[W{last_wid}]({last_count}人>3人)")
        for wid in occupied:
            if 1 < wid < last_wid:
                count = int(window_people_dict.get(wid, 0))
                if count > 2:
                    reasons.append(f"中间排[W{wid}]({count}人>2人)")
    return bool(reasons), " | ".join(reasons) if reasons else "正常"


def draw_final_osd(img, frame_people, vehicle_people, window_people,
                   is_overload, overload_reason):
    """左上角绘制整车信息，左下角单独绘制各车窗人数。"""
    status = "OVERLOAD: YES" if is_overload else "OVERLOAD: NO"
    color = (40, 40, 220) if is_overload else (40, 180, 40)
    breakdown = " ".join(
        f"W{wid}:{int(count)}" for wid, count in sorted(window_people.items()))
    line1 = (f"frame people: {int(frame_people)} | "
             f"vehicle people: {int(vehicle_people)} | {status}")
    top_width = min(img.shape[1], max(760, 12 + 11 * len(line1)))
    cv2.rectangle(img, (0, 0), (top_width, 38), (0, 0, 0), -1)
    cv2.putText(img, line1, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

    if breakdown:
        bottom_width = min(img.shape[1], max(180, 12 + 11 * len(breakdown)))
        bottom_top = max(0, img.shape[0] - 36)
        cv2.rectangle(img, (0, bottom_top),
                      (bottom_width, img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, breakdown, (8, bottom_top + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)


def read_actual_count(image_dir):
    for cand in ("total_registered.txt", "actual_count.txt"):
        p = os.path.join(image_dir, cand)
        if os.path.exists(p):
            try:
                for tok in open(p, encoding="utf-8").read().replace(",", " ").split():
                    if tok.isdigit():
                        return int(tok)
            except Exception as e:
                print(f"[警告] 读取 {cand} 失败: {e}")
    return None


# ==========================================================
# 8. 独立运行
# ==========================================================
def _natural_frame_key(name):
    """Sort numbered frames as 0, 1, 2 ... 10 instead of 0, 1, 10 ... 2."""
    import re
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r"(\d+)", name))


def run_sequence(seq_dir, output_dir, model_path=MODEL_PATH,
                 head_conf=HEAD_CONF, car_conf=CAR_CONF, window_conf=WINDOW_CONF,
                 save_images=True, model=None, sequence_mode=None):
    """跑一个序列，返回计数结果字典。供 CLI 与外部脚本复用。"""
    if model is None:
        model = YOLO(model_path)
    counter = WindowedHeadCounter()
    os.makedirs(output_dir, exist_ok=True)
    pending_draws = []

    images = sorted((f for f in os.listdir(seq_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))),
                    key=_natural_frame_key)
    actual = read_actual_count(seq_dir)
    hard_false_ids = {}
    hard_true_ids = {}
    if OSNET_HARD_FALSE_IDS_FILE and os.path.isfile(OSNET_HARD_FALSE_IDS_FILE):
        try:
            with open(OSNET_HARD_FALSE_IDS_FILE, "r", encoding="utf-8") as f:
                hard_false_ids = json.load(f)
        except Exception as exc:
            print(f"[OSNet hard-negative export] label map read failed: {exc}")
    if OSNET_HARD_TRUE_IDS_FILE and os.path.isfile(OSNET_HARD_TRUE_IDS_FILE):
        try:
            with open(OSNET_HARD_TRUE_IDS_FILE, "r", encoding="utf-8") as f:
                hard_true_ids = json.load(f)
        except Exception as exc:
            print(f"[OSNet hard-positive export] label map read failed: {exc}")
    sequence_false_ids = set(int(x) for x in hard_false_ids.get(
        os.path.basename(os.path.normpath(seq_dir)), []))
    sequence_true_ids = set(int(x) for x in hard_true_ids.get(
        os.path.basename(os.path.normpath(seq_dir)), []))
    hard_export_counts = {}
    # Short evaluation sequences have only a few views; inspect every frame so
    # sparse real tracks are not missed by a regular stride.
    hard_export_step = (1 if len(images) <= 20
                        else max(1, len(images) // 4))

    for i, name in enumerate(images):
        img = imread_cn(os.path.join(seq_dir, name))
        if img is None:
            continue
        img_w = img.shape[1]

        r = model.predict(img, conf=min(head_conf, car_conf, window_conf),
                          iou=NMS_IOU, imgsz=IMGSZ, verbose=False)[0]
        head_dets, head_feats, head_confs, head_person_probs = [], [], [], []
        win_dets, car_dets, context_car_dets = [], [], []
        if r.boxes is not None:
            for box in r.boxes:
                cid = int(box.cls[0].cpu().numpy())
                score = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                if cid == CLS_HEAD:
                    if score < head_conf:
                        continue
                    crop = img[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                        continue
                    feat, person_prob, person_threshold = extract_feat_and_person_prob(crop)
                    head_dets.append((x1, y1, x2, y2))
                    head_feats.append(feat)
                    head_confs.append(score)
                    head_person_probs.append(person_prob)
                elif cid == CLS_WINDOW and score >= window_conf:
                    win_dets.append((x1, y1, x2, y2))
                elif cid == CLS_CAR and score >= CAR_CONTEXT_CONF:
                    car_box = (x1, y1, x2, y2)
                    context_car_dets.append(car_box)
                    if score >= car_conf:
                        car_dets.append(car_box)

        head_dets, head_feats, head_confs, head_person_probs = (
            suppress_near_duplicate_heads(
                head_dets, head_feats, head_confs, head_person_probs))

        labels = counter.update(head_dets, head_feats, head_confs,
                                win_dets, car_dets, img_w, context_car_dets,
                                head_person_probs=head_person_probs)

        render = [f"W{g // WindowedHeadCounter.GID_STRIDE}:{g % WindowedHeadCounter.GID_STRIDE}"
                  if g is not None else "" for g in labels]
        if (OSNET_HARD_EXPORT_DIR and
                (sequence_false_ids or sequence_true_ids) and
                i % hard_export_step == 0):
            for det_i, label in enumerate(labels):
                if label is None or det_i >= len(head_dets):
                    continue
                full_id = int(label)
                if full_id not in sequence_false_ids and full_id not in sequence_true_ids:
                    continue
                category = "non_person" if full_id in sequence_false_ids else "person"
                export_dir = os.path.join(OSNET_HARD_EXPORT_DIR, category)
                os.makedirs(export_dir, exist_ok=True)
                count_key = (category, full_id)
                saved = hard_export_counts.get(count_key, 0)
                if saved >= 4:
                    continue
                x1, y1, x2, y2 = map(int, head_dets[det_i])
                crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue
                out_name = (f"{os.path.basename(os.path.normpath(seq_dir))}_"
                            f"{i:06d}_id{full_id}_{category}_{saved}.jpg")
                cv2.imwrite(os.path.join(export_dir, out_name), crop)
                hard_export_counts[count_key] = saved + 1
        print(f"[{i + 1}/{len(images)}] {name} | 窗 {[w for w, _ in counter.last_windows]} "
              f"| shift={counter.last_shift[0]:.0f},{counter.last_shift[1]:.0f}"
              f"({counter.last_shift_source}) | Σ每窗={counter.total_per_window()} "
              f"整车={counter.total_whole_car()}")
        print(f"      头 {len(head_dets)} 个 | 标签 {render}")
        # 逐个打印本帧有效头肩检测的 YOLO 置信度（不修改检测/跟踪/计数逻辑）
        if head_confs:
            for head_idx, (head_box, head_score) in enumerate(zip(head_dets, head_confs)):
                track_label = render[head_idx] if head_idx < len(render) else ""
                print(f"      [头肩置信度] #{head_idx + 1} "
                      f"conf={head_score:.4f} | ID={track_label or '未分配'} "
                      f"| bbox={tuple(head_box)}")
        else:
            print("      [头肩置信度] 本帧无有效头肩检测")

        if OSNET_SCORE_TRACE:
            for head_idx, head_box in enumerate(head_dets):
                track_label = render[head_idx] if head_idx < len(render) else "unassigned"
                print(f"[OSNet ReID feature] frame={name} ID={track_label} "
                      f"auth_filter=disabled bbox={tuple(head_box)}")

        if save_images:
            assigned = [idx for idx, label in enumerate(render) if label]
            pending_draws.append({
                "name": name,
                "frame_index": i + 1,
                "img": img.copy(),
                "windows": list(counter.last_windows),
                "heads": [head_dets[idx] for idx in assigned],
                "labels": [render[idx] for idx in assigned],
                "confs": [head_confs[idx] for idx in assigned],
                "track_keys": [
                    (int(counter.last_head_window.get(idx, 0)),
                     int(labels[idx]) % WindowedHeadCounter.GID_STRIDE)
                    for idx in assigned
                ],
                "frame_people": sum(1 for label in labels if label is not None),
            })

    final_window_people = counter.per_window_breakdown()
    final_window_diagnostics = counter.per_window_diagnostics()

    # 第一帧尚无位移时，车窗方向只能由车框几何关系暂定；厢式货车等车型
    # 的货厢会显著偏移车辆中心。序列结束后以整段累计运动确认车头方向。
    # 若暂定方向相反，仅在最终输出层整体反转窗号，跟踪状态不重置，避免
    # 中途换窗导致身份重复。
    window_id_map = {int(wid): int(wid) for wid in final_window_people}
    wt = counter.window_tracker
    # The first visible frame can miss the leading window.  In that case the
    # visible windows are provisionally numbered W1/W2 and, when the missing
    # leading window appears, it used to become W3 (for example: W3,W1,W2).
    # Calibrate the output IDs from the last frame having the largest complete
    # set of simultaneous windows.  This changes display/count ownership only;
    # head-track histories and their fixed hit requirements stay untouched.
    complete_frames = [item for item in pending_draws
                       if len(item.get("windows", ())) >= 2]
    if complete_frames:
        max_visible = max(len(item["windows"]) for item in complete_frames)
        reference = next(item for item in reversed(complete_frames)
                         if len(item["windows"]) == max_visible)
        ref_windows = reference["windows"]
        ref_ids = [int(wid) for wid, _ in ref_windows]
        known_ids = {int(wid) for wid in final_window_people}
        if GENERAL_COUNT_TRACE:
            print(f"[window-order reference] sign={wt.motion_sign} "
                  f"ref={[(wid, round(center(box)[0], 1)) for wid, box in ref_windows]} "
                  f"known={sorted(known_ids)}")
        if (len(set(ref_ids)) == len(ref_ids)
                and len(ref_ids) == len(known_ids)
                and set(ref_ids) == known_ids
                and wt.motion_sign is not None):
            physical_order = sorted(
                ref_windows,
                key=lambda item: -int(wt.motion_sign) * center(item[1])[0])
            spatial_map = {int(old_wid): rank + 1
                           for rank, (old_wid, _) in enumerate(physical_order)}
            if any(spatial_map[wid] != wid for wid in spatial_map):
                window_id_map = spatial_map
                final_window_people = {
                    window_id_map.get(int(wid), int(wid)): count
                    for wid, count in final_window_people.items()
                }
                final_window_diagnostics = {
                    window_id_map.get(int(wid), int(wid)): diag
                    for wid, diag in final_window_diagnostics.items()
                }
                for item in pending_draws:
                    item["windows"] = [
                        (window_id_map.get(int(wid), int(wid)), box)
                        for wid, box in item["windows"]
                    ]
                if GENERAL_COUNT_TRACE:
                    print(f"[window-order canonicalize] map={spatial_map}")
    confirmed_sign = 0
    # 使用逐帧位移的中位数确认方向，避免主车在末帧出画后误跳到其他车辆，
    # 单个巨大反向位移把整段累计方向翻转。
    motion_samples = list(getattr(wt, "motion_dx_history", ()))
    resolved_sequence_mode, sequence_profile = resolve_sequence_mode(
        sequence_mode, len(images), motion_samples)
    is_dense_sequence = resolved_sequence_mode == "dense"
    # If adjacent windows have reciprocal count drift while the vehicle total
    # exactly equals the sum of their simultaneous occupancy peaks, correct
    # ownership to those peaks. This repairs cross-window handoff without
    # changing the total or overriding genuine temporal additions.
    if is_dense_sequence and len(final_window_people) == 2:
        _wids = sorted(final_window_people)
        _peaks = {wid: int(final_window_diagnostics.get(wid, {}).get("peak", 0))
                  for wid in _wids}
        _diffs = {wid: int(final_window_people[wid]) - _peaks[wid]
                  for wid in _wids}
        if (sum(final_window_people.values()) == sum(_peaks.values()) and
                sorted(_diffs.values()) == [-1, 1]):
            final_window_people = dict(_peaks)
            for _wid in _wids:
                if _wid in final_window_diagnostics:
                    final_window_diagnostics[_wid]["general"] = _peaks[_wid]
                    final_window_diagnostics[_wid]["cross_window_peak_rebalance"] = 1
    print(f"[序列模式] requested={sequence_profile['requested']} "
          f"resolved={resolved_sequence_mode} images={len(images)} "
          f"motion_samples={sequence_profile['motion_samples']} "
          f"coverage={sequence_profile['motion_coverage']:.2f} "
          f"median_abs_dx={sequence_profile['median_abs_dx']}")
    # 最终方向只用于决定是否整体反转 W1..Wn。车辆框和车窗框都可能
    # 突变，因此必须由至少两类独立来源共同支持，且加权方向一致率达到
    # 75%。方向证据不足时保持 confirmed_sign=0，不执行破坏性的整体反转。
    direction_evidence = list(getattr(wt, "direction_evidence", ()))
    valid_direction_evidence = [
        (str(source), float(dx), float(weight))
        for source, dx, weight in direction_evidence
        if np.isfinite(dx) and 2.0 < abs(float(dx)) <= 350.0
    ]
    if len(valid_direction_evidence) >= 3:
        positive_weight = sum(weight for _, dx, weight in valid_direction_evidence
                              if dx > 0.0)
        negative_weight = sum(weight for _, dx, weight in valid_direction_evidence
                              if dx < 0.0)
        total_weight = positive_weight + negative_weight
        candidate_sign = 1 if positive_weight > negative_weight else -1
        winning_weight = max(positive_weight, negative_weight)
        agreeing_sources = {
            source for source, dx, _ in valid_direction_evidence
            if dx * candidate_sign > 0.0
        }
        agreeing_distance = sum(
            abs(dx) for _, dx, _ in valid_direction_evidence
            if dx * candidate_sign > 0.0)
        if (total_weight > 0.0 and
                winning_weight / total_weight >= 0.75 and
                len(agreeing_sources) >= 2 and
                agreeing_distance >= _HU_TRAVEL_MIN_DX):
            confirmed_sign = candidate_sign
        if GENERAL_COUNT_TRACE:
            print(f"[最终方向多源确认] evidence={len(valid_direction_evidence)} "
                  f"right_w={positive_weight:.1f} left_w={negative_weight:.1f} "
                  f"sources={sorted(agreeing_sources)} sign={confirmed_sign}")
    if (confirmed_sign and wt.motion_sign is not None
            and confirmed_sign != int(wt.motion_sign)):
        known_wids = sorted(window_id_map)
        window_id_map = dict(zip(known_wids, reversed(known_wids)))
        final_window_people = {
            window_id_map.get(int(wid), int(wid)): count
            for wid, count in final_window_people.items()
        }
        final_window_diagnostics = {
            window_id_map.get(int(wid), int(wid)): diag
            for wid, diag in final_window_diagnostics.items()
        }
        for item in pending_draws:
            item["windows"] = [
                (window_id_map.get(int(wid), int(wid)), box)
                for wid, box in item["windows"]
            ]
    # Re-estimate the physical W1 direction from a stable tail layout.  Early
    # opposing/background traffic can dominate the first car-box motion samples
    # and lock WindowIdTracker.motion_sign to the wrong direction.  Require at
    # least two adjacent tail-frame pairs, each supported by two or more same-ID
    # windows moving coherently with stable scale; this avoids single-box jumps.
    tail_window_pairs = []
    tail_support = {}
    tail_start_frame = max(2, int(np.ceil(len(images) * 0.5)))
    for prev_item, curr_item in zip(pending_draws, pending_draws[1:]):
        if int(curr_item.get("frame_index", 0)) < tail_start_frame:
            continue
        prev_by_id = {int(wid): box for wid, box in prev_item.get("windows", ())}
        curr_by_id = {int(wid): box for wid, box in curr_item.get("windows", ())}
        shared_ids = sorted(set(prev_by_id) & set(curr_by_id))
        if len(shared_ids) < 2:
            continue
        deltas = []
        size_ok = True
        for wid in shared_ids:
            a, b = prev_by_id[wid], curr_by_id[wid]
            aw, ah = max(1.0, float(a[2] - a[0])), max(1.0, float(a[3] - a[1]))
            bw, bh = max(1.0, float(b[2] - b[0])), max(1.0, float(b[3] - b[1]))
            if (min(aw, bw) / max(aw, bw) < 0.45 or
                    min(ah, bh) / max(ah, bh) < 0.45 or
                    abs(center(a)[1] - center(b)[1]) > max(90.0, 1.2 * max(ah, bh))):
                size_ok = False
                break
            deltas.append(float(center(b)[0] - center(a)[0]))
        if not size_ok:
            continue
        med_dx = float(np.median(deltas))
        mad_dx = float(np.median(np.abs(np.asarray(deltas, dtype=np.float32) - med_dx)))
        agreeing = sum(1 for dx in deltas if dx * med_dx > 0.0 and abs(dx) > 2.0)
        if (2.0 < abs(med_dx) <= 350.0 and agreeing / len(deltas) >= 0.80 and
                mad_dx <= max(35.0, 0.35 * abs(med_dx))):
            tail_window_pairs.append((int(curr_item.get("frame_index", 0)), med_dx,
                                      tuple(shared_ids), curr_by_id))
            for wid in shared_ids:
                tail_support[wid] = tail_support.get(wid, 0) + 1
    tail_stable_ids = {wid for wid, support in tail_support.items() if support >= 2}
    tail_signs = [1 if dx > 0.0 else -1 for _, dx, _, _ in tail_window_pairs]
    tail_order_sign = 0
    tail_reference_layout = None
    if len(tail_window_pairs) >= 2 and len(tail_stable_ids) >= 2:
        candidate_tail_sign = 1 if sum(tail_signs) > 0 else -1
        if (sum(1 for sign in tail_signs if sign == candidate_tail_sign) /
                len(tail_signs) >= 0.80 and
                sum(abs(dx) for _, dx, _, _ in tail_window_pairs
                    if dx * candidate_tail_sign > 0.0) >= _HU_TRAVEL_MIN_DX):
            tail_order_sign = candidate_tail_sign
            latest_layout = next(
                (by_id for _, _, _, by_id in reversed(tail_window_pairs)
                 if tail_stable_ids.issubset(by_id)), None)
            if latest_layout is not None:
                current_ids = set(int(wid) for wid in window_id_map.values())
                current_ids.update(int(wid) for wid in final_window_people)
                current_ids.update(int(wid) for item in pending_draws
                                   for wid, _ in item.get("windows", ()))
                ordered_stable = sorted(
                    tail_stable_ids,
                    key=lambda wid: -tail_order_sign * center(latest_layout[wid])[0])
                relabel = {wid: rank + 1 for rank, wid in enumerate(ordered_stable)}
                tail_reference_layout = [
                    (relabel[wid], latest_layout[wid]) for wid in ordered_stable
                ]
                for wid in sorted(current_ids - tail_stable_ids):
                    relabel[wid] = len(ordered_stable) + 1 + sum(
                        1 for old in current_ids - tail_stable_ids if old < wid)
                old_to_current = dict(window_id_map)
                inverse_current = {current: old for old, current in old_to_current.items()}
                final_window_people = {
                    relabel.get(int(wid), int(wid)): count
                    for wid, count in final_window_people.items()
                }
                final_window_diagnostics = {
                    relabel.get(int(wid), int(wid)): diag
                    for wid, diag in final_window_diagnostics.items()
                }
                for item in pending_draws:
                    item["windows"] = [
                        (relabel.get(int(wid), int(wid)), box)
                        for wid, box in item.get("windows", ())
                    ]
                window_id_map = {
                    old: relabel.get(current, current)
                    for old, current in old_to_current.items()
                }
                if GENERAL_COUNT_TRACE:
                    print(f"[尾段车窗方向重校准] sign={tail_order_sign} "
                          f"pairs={[(frame, round(dx, 1), ids) for frame, dx, ids, _ in tail_window_pairs]} "
                          f"stable_ids={sorted(tail_stable_ids)} relabel={relabel}")

    # Per-frame presentation repair: if a frame contains a complete contiguous
    # set W1..Wn, its labels must follow the physical front-to-rear order in
    # that frame.  This handles a leading-window detector dropout that appears
    # one frame later, without compacting genuine suffixes such as W2/W3 after
    # W1 has already left the image.
    first_complete_index = None
    sequence_max_windows = max(
        (len(item.get("windows", ())) for item in pending_draws), default=0)
    if tail_order_sign and tail_reference_layout:
        # Prefer the stable main-car tail layout over early frames polluted by
        # an opposite-direction background vehicle.
        sequence_max_windows = len(tail_reference_layout)
    for _item in pending_draws:
        if len(_item.get("windows", ())) == sequence_max_windows:
            first_complete_index = int(_item.get("frame_index", 0))
            break
    # Keep the most recent complete physical layout as a geometric anchor.
    # When an edge window is missed later, decide whether the visible windows
    # are a prefix (W1...) or suffix (...Wn) from their relative spacing.  A
    # mere drop in detection count must not automatically mean that W1 left.
    latest_complete_ordered = (
        list(tail_reference_layout) if tail_order_sign and tail_reference_layout else None
    )
    for item in pending_draws:
        frame_windows = item.get("windows", [])
        frame_ids = {int(wid) for wid, _ in frame_windows}
        n_frame = len(frame_windows)
        frame_map = {}
        display_motion_sign = tail_order_sign or confirmed_sign or wt.motion_sign
        if (n_frame >= 2 and len(frame_ids) == n_frame
                and frame_ids == set(range(1, n_frame + 1))
                and display_motion_sign is not None):
            ordered = sorted(
                frame_windows,
                key=lambda pair: -int(display_motion_sign) * center(pair[1])[0])
            if n_frame == sequence_max_windows and not (tail_order_sign and tail_reference_layout):
                latest_complete_ordered = list(ordered)
            rank_offset = 0
            if (latest_complete_ordered is not None and
                    2 <= n_frame < sequence_max_windows):
                curr_centers = [center(box)[0] for _, box in ordered]
                curr_gaps = [abs(curr_centers[j + 1] - curr_centers[j])
                             for j in range(n_frame - 1)]
                candidate_scores = []
                for offset in range(sequence_max_windows - n_frame + 1):
                    ref_slice = latest_complete_ordered[offset:offset + n_frame]
                    ref_centers = [center(box)[0] for _, box in ref_slice]
                    ref_gaps = [abs(ref_centers[j + 1] - ref_centers[j])
                                for j in range(n_frame - 1)]
                    gap_errors = [
                        abs(np.log(max(cg, 1.0) / max(rg, 1.0)))
                        for cg, rg in zip(curr_gaps, ref_gaps)]
                    candidate_scores.append((float(np.mean(gap_errors)), offset))
                candidate_scores.sort()
                best_score, best_offset = candidate_scores[0]
                # Require a real geometric advantage.  Ambiguous layouts keep
                # the tracker's existing IDs instead of shifting every window.
                second_score = (candidate_scores[1][0]
                                if len(candidate_scores) > 1 else float("inf"))
                if best_score + 0.12 < second_score:
                    rank_offset = int(best_offset)
                if GENERAL_COUNT_TRACE:
                    print(f"[partial-window-layout] {item['name']} "
                          f"scores={[(round(s, 3), o) for s, o in candidate_scores]} "
                          f"offset={rank_offset}")
            frame_map = {int(old_wid): rank + 1 + rank_offset
                         for rank, (old_wid, _) in enumerate(ordered)}
            if any(frame_map[wid] != wid for wid in frame_map):
                item["windows"] = [
                    (frame_map.get(int(wid), int(wid)), box)
                    for wid, box in frame_windows
                ]
                if GENERAL_COUNT_TRACE:
                    print(f"[frame-window-order] {item['name']} map={frame_map}")
        item["frame_window_id_map"] = frame_map
    final_window_people = dict(sorted(final_window_people.items()))
    final_window_diagnostics = dict(sorted(final_window_diagnostics.items()))

    # Dense tail-frame cross-window handoff: near the image boundary the same
    # head can be reassigned from the preceding window to a newly created last
    # window for only two frames.  Per-window trackers then count it twice even
    # though the independent whole-car tracker keeps one identity.  Reconcile
    # only this narrow lifecycle (one-person last window, <=2 observations,
    # exactly one-count disagreement) so genuine short detections remain valid
    # when the whole-car identity space also sees them as new people.
    provisional_people = sum(int(v) for v in final_window_people.values())
    whole_people = counter.total_whole_car()
    if (is_dense_sequence and whole_people is not None and
            provisional_people - int(whole_people) == 1 and
            final_window_people):
        last_wid = max(int(wid) for wid in final_window_people)
        old_last_wids = [int(old_wid) for old_wid, new_wid in window_id_map.items()
                         if int(new_wid) == last_wid]
        old_last_wid = old_last_wids[0] if len(old_last_wids) == 1 else last_wid
        last_tracker = counter.window_head_trackers.get(old_last_wid)
        raw_tracker = getattr(last_tracker, "_t", last_tracker)
        last_summaries = getattr(raw_tracker, "track_summaries", {}) or {}
        last_track_lengths = [len(set(s.get("frames", ())))
                              for s in last_summaries.values()]
        if (int(final_window_people.get(last_wid, 0)) == 1 and
                last_track_lengths and max(last_track_lengths) <= 2):
            final_window_people[last_wid] = 0
            if last_wid in final_window_diagnostics:
                final_window_diagnostics[last_wid]["dense_cross_window_dedupe"] = 1
            if GENERAL_COUNT_TRACE:
                print(f"[密集帧跨窗去重] W{last_wid} -1 "
                      f"per-window={provisional_people} whole-car={whole_people} "
                      f"track_lengths={last_track_lengths}")

    # Long dense sequences sometimes contain a single sustained passenger plus
    # several short, weak reflections that satisfy the fixed hit threshold.
    # Do not relax that threshold globally.  Instead, use the whole lifecycle:
    # exactly one 30+ observation track, at least four competing fragments, and
    # every competitor both short (<=12 observations) and never confidently
    # head-like.  This preserves ordinary short passengers unless the sequence
    # exhibits the complete dominant-track pattern.
    dense_track_records = []
    for old_wid, tracker in counter.window_head_trackers.items():
        raw_tracker = getattr(tracker, "_t", tracker)
        for summary in (getattr(raw_tracker, "track_summaries", {}) or {}).values():
            observations = summary.get("observations", {}) or {}
            if not observations:
                continue
            dense_track_records.append({
                "old_wid": int(old_wid),
                "length": len(set(summary.get("frames", ()))),
                "max_conf": max(float(o.get("conf", 0.0))
                                for o in observations.values()),
            })
    # Preserve the historical >=100-frame thresholds. For shorter full
    # sequences use proportional lifecycle thresholds, never changing the
    # normal counter's fixed two/three-frame validity rules.
    if len(images) >= 100:
        dense_dominant_min, dense_fragment_max = 30, 12
    else:
        dense_dominant_min = max(12, int(np.ceil(len(images) * 0.25)))
        dense_fragment_max = max(4, int(np.ceil(len(images) * 0.12)))
    dominant_tracks = [r for r in dense_track_records
                       if r["length"] >= dense_dominant_min]
    if (is_dense_sequence and len(images) >= 40 and
            len(dense_track_records) >= 5 and
            len(dominant_tracks) == 1):
        dominant = dominant_tracks[0]
        competitors = [r for r in dense_track_records if r is not dominant]
        if (competitors and all(r["length"] <= dense_fragment_max and
                               r["max_conf"] <= 0.60
                               for r in competitors)):
            dominant_wid = int(window_id_map.get(dominant["old_wid"],
                                                  dominant["old_wid"]))
            for wid in list(final_window_people):
                final_window_people[wid] = 1 if int(wid) == dominant_wid else 0
            if GENERAL_COUNT_TRACE:
                print(f"[密集长序列弱碎轨过滤] keep=W{dominant_wid} "
                      f"length={dominant['length']} fragments="
                      f"{[(r['old_wid'], r['length'], round(r['max_conf'], 3)) for r in competitors]}")
    final_people = sum(int(v) for v in final_window_people.values())
    # Dense-sequence window blind spot: accept one missing passenger only when
    # three independent facts agree exactly: the simultaneous detection peak and
    # the whole-car stable identity count are equal, and both exceed the
    # per-window result by exactly one.  The in-window simultaneous peak must
    # independently support the whole-car count.  This preserves fixed track hit requirements and
    # does not use a vehicle type or a configured seat capacity.
    diag_summary = counter.diag_stats()
    if (is_dense_sequence and whole_people is not None and
            int(whole_people) == final_people + 1 and
            int(diag_summary.get("peak_det", -1)) == int(whole_people) and
            int(diag_summary.get("peak_in_car", -1)) == int(whole_people) and
            int(diag_summary.get("peak_in_car", -1)) ==
            int(diag_summary.get("peak_in_win", -1)) + 1 and
            int(diag_summary.get("peak_in_win", -1)) == final_people and
            final_window_people):
        positive_windows = [wid for wid, count in final_window_people.items()
                            if int(count) > 0]
        if positive_windows:
            blind_wid = min(positive_windows,
                            key=lambda wid: (int(final_window_people[wid]), int(wid)))
            final_window_people[blind_wid] = int(final_window_people[blind_wid]) + 1
            final_people += 1
            final_window_diagnostics.setdefault(blind_wid, {})[
                "dense_window_blind_spot_recovery"] = 1
            if GENERAL_COUNT_TRACE:
                print(f"[密集帧车窗盲区补偿] W{blind_wid} +1 "
                      f"per-window={final_people - 1} whole-car={whole_people} "
                      f"diag={diag_summary}")
    is_overload, overload_reason = check_overload(final_window_people)

    # general_count() 已完成碎轨合并；绘图前把早先缓存的原始轨迹号统一
    # 替换为物理身份的规范 ID，例如车辆运动导致的 102 -> 103 显示回 102。
    canonical_ids = {}
    suffix_ids = {}
    frame_ids = {}
    rejected_track_keys = set()
    rejected_observation_keys = set()
    for wid, tracker in counter.window_head_trackers.items():
        canonical_ids.update(getattr(tracker, "last_general_id_map", {}))
        suffix_ids.update(getattr(tracker, "last_general_suffix_map", {}))
        frame_ids.update(getattr(tracker, "last_general_frame_id_map", {}))
        rejected_track_keys.update(
            (int(wid), int(tid))
            for tid in getattr(tracker, "last_suppressed_track_ids", set())
        )
        rejected_observation_keys.update(
            (int(frame_no), int(wid), int(tid))
            for frame_no, tid in getattr(
                tracker, "last_general_rejected_observations", set())
        )
    for item in pending_draws:
        if rejected_track_keys or rejected_observation_keys:
            item_frame = int(item.get("frame_index", 0))
            kept = [i for i, key in enumerate(item.get("track_keys", ()))
                    if (tuple(key) not in rejected_track_keys and
                        (item_frame, int(key[0]), int(key[1]))
                        not in rejected_observation_keys)]
            item["heads"] = [item["heads"][i] for i in kept]
            item["labels"] = [item["labels"][i] for i in kept]
            item["confs"] = [item["confs"][i] for i in kept]
            item["track_keys"] = [item["track_keys"][i] for i in kept]
            item["frame_people"] = len(kept)
        frame_window_id_map = item.get("frame_window_id_map", {})
        remapped = []
        for label in item["labels"]:
            try:
                prefix, raw_id = label.rsplit(":", 1)
                raw_id = int(raw_id)
                raw_id = int(frame_ids.get(
                    (int(item.get("frame_index", 0)), raw_id), raw_id))
                suffix = suffix_ids.get(raw_id)
                if (suffix is not None and
                        int(item.get("frame_index", 0)) >= int(suffix[0])):
                    raw_id = int(suffix[1])
                shown_id = canonical_ids.get(raw_id, raw_id)
                old_wid, local_id = divmod(int(shown_id), 100)
                shown_id = window_id_map.get(old_wid, old_wid) * 100 + local_id
                shown_wid, shown_local_id = divmod(int(shown_id), 100)
                shown_id = (frame_window_id_map.get(shown_wid, shown_wid) * 100
                            + shown_local_id)
                remapped.append(f"{prefix}:{shown_id}")
            except (AttributeError, ValueError):
                remapped.append(label)
        item["labels"] = remapped

    # 与参考脚本一致：所有帧统一显示该车辆最终人数和最终超员状态。
    if save_images:
        for item in pending_draws:
            draw_windows(item["img"], item["windows"])
            draw_boxes_with_labels(item["img"], item["heads"],
                                   item["labels"], confs=item["confs"])
            draw_final_osd(item["img"], item["frame_people"], final_people,
                           final_window_people, is_overload, overload_reason)
            if actual is not None:
                cv2.putText(item["img"], f"Actual: {actual}", (20, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imwrite(os.path.join(output_dir, item["name"]), item["img"])

    res = {
        "seq": os.path.basename(os.path.normpath(seq_dir)),
        "per_window": final_people,
        "per_window_raw": counter.total_per_window_raw(),
        "whole_car": counter.total_whole_car(),
        "per_window_breakdown": final_window_people,
        "count_diagnostics": final_window_diagnostics,
        "is_overload": is_overload,
        "overload_reason": overload_reason,
        "actual": actual,
        "diag": counter.diag_stats(),
        "sequence_profile": sequence_profile,
    }
    print("=" * 56)
    print(f" {res['seq']}")
    print(f"  通用融合人数   : {res['per_window']}   {res['per_window_breakdown']}")
    print(f"  原始 distinct  : {res['per_window_raw']}")
    print(f"  计数诊断       : {res['count_diagnostics']}")
    print(f"  整车 distinct  : {res['whole_car']}")
    print(f"  序列模式        : {resolved_sequence_mode} | {sequence_profile}")
    print(f"  超员判断        : {'是' if is_overload else '否'} | {overload_reason}")
    if actual is not None:
        print(f"  人工核实       : {actual}")
    print("=" * 56)
    return res


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _contains_images(folder):
    return any(f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
               for f in __import__("pathlib").Path(folder).iterdir())


def discover_sequences(input_dir, output_root):
    """单车目录或多车根目录；不把结果目录误当输入。"""
    from pathlib import Path
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {root}")
    out = Path(output_root).expanduser().resolve()
    if _contains_images(root):
        return [root]
    sequences = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        resolved = child.resolve()
        if resolved == out or resolved in out.parents or out in resolved.parents:
            continue
        if _contains_images(child):
            sequences.append(child)
    return sequences


def _save_batch_summary(results, output_root):
    """每完成一辆车写一次，避免中途停止丢失已有统计。"""
    from pathlib import Path
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "batch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    csv_path = out / "batch_results.csv"
    fields = ["seq", "people", "per_window", "per_window_raw", "whole_car",
              "whole_car_raw", "W1", "W2", "W3",
              "per_window_breakdown", "count_diagnostics", "is_overload",
              "overload_reason", "actual", "diag", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            wins = r.get("per_window_breakdown") or {}
            def get_win(wid):
                return wins.get(wid, wins.get(str(wid), 0))
            final_people = r.get("per_window", "")
            writer.writerow({
                "seq": r.get("seq", ""),
                # CSV has one authoritative people count: exactly the value
                # drawn by draw_final_osd().  Keep the independent whole-car
                # tracker only as an explicitly named raw diagnostic column.
                "people": final_people,
                "per_window": final_people,
                "per_window_raw": r.get("per_window_raw", ""),
                "whole_car": final_people,
                "whole_car_raw": r.get("whole_car", ""),
                "W1": get_win(1), "W2": get_win(2), "W3": get_win(3),
                "per_window_breakdown": json.dumps(wins, ensure_ascii=False),
                "count_diagnostics": json.dumps(r.get("count_diagnostics", {}), ensure_ascii=False),
                "is_overload": r.get("is_overload", ""),
                "overload_reason": r.get("overload_reason", ""),
                "actual": r.get("actual"),
                "diag": json.dumps(r.get("diag", {}), ensure_ascii=False),
                "error": r.get("error", ""),
            })


def run_batch(input_dir=INPUT_DIR, output_root=OUTPUT_DIR,
              model_path=MODEL_PATH, head_conf=HEAD_CONF,
              car_conf=CAR_CONF, window_conf=WINDOW_CONF,
              save_images=SAVE_ANNOTATED_IMAGES,
              continue_on_error=CONTINUE_ON_ERROR,
              sequence_mode=None):
    """加载一次 YOLO，逐车重建 Counter，防止跨车辆 ID 串联。"""
    from pathlib import Path
    sequences = discover_sequences(input_dir, output_root)
    if not sequences:
        raise ValueError(f"未发现图片或包含图片的车辆子目录: {input_dir}")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    print(f"[批量] 共发现 {len(sequences)} 辆车，加载 YOLO: {model_path}")
    model = YOLO(model_path)
    results = []
    for i, seq in enumerate(sequences, 1):
        print(f"\n{'=' * 70}\n[批量 {i}/{len(sequences)}] {seq.name}\n{'=' * 70}")
        out = Path(output_root) / seq.name
        try:
            res = run_sequence(str(seq), str(out), model_path=model_path,
                               head_conf=head_conf, car_conf=car_conf,
                               window_conf=window_conf,
                               save_images=save_images, model=model,
                               sequence_mode=sequence_mode)
            results.append(res)
        except Exception as exc:
            print(f"[错误] {seq.name}: {exc}")
            traceback.print_exc()
            results.append({"seq": seq.name, "error": str(exc)})
            _save_batch_summary(results, output_root)
            if not continue_on_error:
                raise
        _save_batch_summary(results, output_root)
    ok = [r for r in results if "error" not in r]
    print(f"\n[批量完成] 成功 {len(ok)}/{len(sequences)}，结果: {output_root}")
    return results


def _main():
    ap = argparse.ArgumentParser(description="sideview 多车辆目录批量车窗计数")
    ap.add_argument("seq_dir", nargs="?", default=INPUT_DIR,
                    help="单车目录或包含多辆车子目录的根目录")
    ap.add_argument("output_dir", nargs="?", default=OUTPUT_DIR,
                    help="输出根目录")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--head-conf", type=float, default=HEAD_CONF)
    ap.add_argument("--car-conf", type=float, default=CAR_CONF)
    ap.add_argument("--window-conf", type=float, default=WINDOW_CONF)
    ap.add_argument("--no-images", action="store_true", help="不保存标注图")
    ap.add_argument("--sequence-mode", choices=("auto", "short", "dense"),
                    default=SEQUENCE_MODE,
                    help="序列模式；默认 auto 自动兼容8张图和完整密集序列")
    a = ap.parse_args()
    run_batch(a.seq_dir, a.output_dir, a.model, a.head_conf,
              a.car_conf, a.window_conf,
              save_images=SAVE_ANNOTATED_IMAGES and not a.no_images,
              sequence_mode=a.sequence_mode)


if __name__ == "__main__":
    _main()
