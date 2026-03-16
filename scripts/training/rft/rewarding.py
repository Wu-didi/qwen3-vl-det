"""RFT 奖励函数与解析辅助工具。"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

NO_DETECTION_PATTERNS = [
    "未检测到相关设备",
    "未检测到设备",
    "未检测到目标",
    "没有检测到",
    "no relevant equipment",
    "no equipment detected",
]

BOX_PATTERN = (
    r"<box>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*</box>"
)

ANOMALY_KEYWORDS = ["异常", "全灭", "损坏", "故障", "破损", "不亮", "错误", "黑屏", "全亮"]

CATEGORY_ALIASES = {
    "traffic_signal": "traffic_signal",
    "交通信号灯": "traffic_signal",
    "信号灯": "traffic_signal",
    "guidance_screen": "guidance_screen",
    "交通诱导屏": "guidance_screen",
    "诱导屏": "guidance_screen",
    "height_limit_bar": "height_limit_bar",
    "限高架": "height_limit_bar",
    "cabinet": "cabinet",
    "机柜": "cabinet",
    "backpack_box": "backpack_box",
    "背包箱": "backpack_box",
}

NON_ANOMALY_STATES = {"normal", "正常", "unknown"}


@dataclass
class ParsedDetection:
    """从模型文本中解析出的结构化检测结果。"""

    category: str
    status: str
    is_anomaly: bool
    bbox: List[int]


@dataclass
class RiskRewardConfig:
    """风险感知奖励的超参数配置。"""

    match_iou_threshold: float = 0.5
    hallucination_unit_penalty: float = 0.35
    no_detection_missing_penalty: float = 0.2
    omission_penalty: float = 1.0


RISK_REWARD_CFG = RiskRewardConfig()


def set_risk_reward_config(cfg: RiskRewardConfig) -> None:
    """更新全局风险奖励配置（供各奖励函数共享）。"""
    global RISK_REWARD_CFG
    RISK_REWARD_CFG = cfg


def get_risk_reward_config() -> RiskRewardConfig:
    """返回当前风险奖励配置。"""
    return RISK_REWARD_CFG


def _is_no_detection_response(text: str) -> bool:
    """判断模型是否明确输出“未检测到设备”语义。"""
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in NO_DETECTION_PATTERNS):
        return True

    payload = _extract_json_payload(text)
    return isinstance(payload, dict) and payload.get("detections") == []


def _has_structured_detection_format(text: str) -> bool:
    """检查输出是否满足“序号 + 状态 + 坐标框”的结构化格式。"""
    has_box = bool(re.search(BOX_PATTERN, text))
    has_numbered = bool(re.search(r"\d+\.\s+\S+", text))
    has_status = bool(re.search(r"状态[：:]\s*\S+", text))
    return has_box and has_numbered and has_status


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    """从纯文本或 markdown 代码块中提取第一个 JSON 对象。"""
    if not text:
        return None

    clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    if not clean:
        return None

    try:
        payload = json.loads(clean)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_category_alias(category: str) -> str:
    normalized = re.sub(r"\s+", "", category.strip().lower())
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]

    for alias, canonical in CATEGORY_ALIASES.items():
        if alias in category or alias in normalized:
            return canonical
    return normalized


def _build_detection_from_json(det: Dict[str, Any]) -> Optional[ParsedDetection]:
    bbox = det.get("bbox_1000")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        bbox = [int(float(v)) for v in bbox]
    except (TypeError, ValueError):
        return None

    category = str(det.get("device_type", "unknown")).strip() or "unknown"
    status = str(det.get("state", "normal")).strip() or "normal"
    return ParsedDetection(
        category=category,
        status=status,
        is_anomaly=_is_anomaly_status(status),
        bbox=bbox,
    )


def _extract_detections_from_ground_truth(ground_truth: Optional[Dict[str, Any]]) -> List[ParsedDetection]:
    if not isinstance(ground_truth, dict):
        return []

    detections: List[ParsedDetection] = []
    for det in ground_truth.get("detections", []):
        if not isinstance(det, dict):
            continue
        parsed = _build_detection_from_json(det)
        if parsed is not None:
            detections.append(parsed)
    return detections


def _get_gt_detections(
    idx: int,
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
) -> List[ParsedDetection]:
    if ground_truth and idx < len(ground_truth):
        gt_item = ground_truth[idx] if isinstance(ground_truth[idx], dict) else None
        parsed = _extract_detections_from_ground_truth(gt_item)
        if parsed or (isinstance(gt_item, dict) and gt_item.get("detections") == []):
            return parsed

    if assistant and idx < len(assistant):
        return _extract_detections(assistant[idx])

    return []


def _get_gt_response_text(idx: int, assistant: Optional[List[str]] = None) -> Optional[str]:
    if assistant and idx < len(assistant):
        return assistant[idx]
    return None


def _is_format_valid(completion: str, gt_response: Optional[str] = None) -> bool:
    """奖励函数统一使用的严格格式门控。"""
    payload = _extract_json_payload(completion)
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        return True

    if _has_structured_detection_format(completion):
        return True

    if gt_response is None:
        return False

    gt_boxes = _extract_boxes(gt_response)
    if not gt_boxes and _is_no_detection_response(completion):
        return True

    return False


def format_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """格式奖励（门控项）：合法=1，不合法=0。"""
    rewards = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)
        if _is_format_valid(completion, gt_response) or (not gt_dets and _is_no_detection_response(completion)):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def bbox_iou_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """基于 IoU 的定位奖励。"""
    rewards = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_boxes = _extract_boxes(completion)
        gt_boxes = [det.bbox for det in _get_gt_detections(idx, assistant, ground_truth)]

        if not gt_boxes:
            if not pred_boxes and _is_no_detection_response(completion):
                reward = 1.0
            else:
                reward = 0.0
        elif not pred_boxes:
            reward = 0.0
        else:
            ious = []
            for gt_box in gt_boxes:
                best_iou = 0.0
                for pred_box in pred_boxes:
                    iou = _compute_iou(pred_box, gt_box)
                    best_iou = max(best_iou, iou)
                ious.append(best_iou)
            reward = sum(ious) / len(ious) if ious else 0.0

        rewards.append(reward)

    return rewards


def category_match_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """类别匹配奖励。"""
    rewards = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_cats = _extract_categories(completion)
        gt_cats = [det.category for det in _get_gt_detections(idx, assistant, ground_truth)]

        if not gt_cats:
            reward = 1.0 if (not pred_cats and _is_no_detection_response(completion)) else 0.0
        elif not pred_cats:
            reward = 0.0
        else:
            matches = 0
            for gt_cat in gt_cats:
                for pred_cat in pred_cats:
                    if _category_match(pred_cat, gt_cat):
                        matches += 1
                        break
            reward = matches / len(gt_cats)

        rewards.append(reward)

    return rewards


def status_accuracy_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """状态准确率奖励（正常/异常）。"""
    rewards = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_statuses = _extract_statuses(completion)
        gt_statuses = [det.status for det in _get_gt_detections(idx, assistant, ground_truth)]

        if not gt_statuses:
            reward = 1.0 if (not pred_statuses and _is_no_detection_response(completion)) else 0.0
        elif not pred_statuses:
            reward = 0.0
        else:
            pred_has_anomaly = any(_is_anomaly_status(s) for s in pred_statuses)
            gt_has_anomaly = any(_is_anomaly_status(s) for s in gt_statuses)
            reward = 1.0 if pred_has_anomaly == gt_has_anomaly else 0.0

        rewards.append(reward)

    return rewards


def set_f1_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """集合级检测 F1 奖励（基于一对一 TP/FP/FN 匹配）。"""
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)

        if gt_count == 0:
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                rewards.append(0.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=True,
        )
        tp = len(matches)
        fp = max(pred_count - tp, 0)
        fn = max(gt_count - tp, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # 使用 F2 score (beta=2)，recall 权重是 precision 的 4 倍
        # F_beta = (1+beta^2)*P*R / (beta^2*P + R)
        rewards.append(_safe_fbeta(precision, recall, beta=2.0))

    return rewards


def localization_quality_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """定位质量奖励：对匹配对计算 IoU 并归一化。"""
    rewards: List[float] = []
    iou_thr = RISK_REWARD_CFG.match_iou_threshold
    denom = max(1e-6, 1.0 - iou_thr)

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)

        if not gt_dets:
            if not pred_dets and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif not pred_dets:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                rewards.append(0.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=iou_thr,
            require_category=True,
        )

        if not matches:
            rewards.append(0.0)
            continue

        mean_iou = sum(m["iou"] for m in matches) / len(matches)
        norm_iou = max(0.0, min(1.0, (mean_iou - iou_thr) / denom))
        rewards.append(norm_iou)

    return rewards


def count_alignment_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """计数一致性奖励：惩罚明显过检和漏检。"""
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_count = len(_extract_detections(completion))
        gt_count = len(_get_gt_detections(idx, assistant, ground_truth))

        if gt_count == 0:
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-penalty)
            continue

        if _is_no_detection_response(completion) and pred_count == 0:
            rewards.append(-min(1.0, RISK_REWARD_CFG.omission_penalty))
            continue

        # 非对称惩罚：漏检（pred < gt）惩罚 2x 于过检（pred > gt）
        diff = pred_count - gt_count
        if diff < 0:
            # 漏检：惩罚力度加倍
            ratio_err = abs(diff) / max(gt_count, 1) * 2.0
        else:
            # 过检：正常惩罚
            ratio_err = diff / max(gt_count, 1)
        rewards.append(max(-1.0, min(1.0, 1.0 - ratio_err)))

    return rewards


def risk_control_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """风险控制奖励：惩罚空场景误报与正样本漏检。"""
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)
        no_detection = _is_no_detection_response(completion)

        if gt_count == 0:
            if pred_count == 0 and no_detection:
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                halluc_penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-halluc_penalty)
            continue

        if pred_count == 0:
            rewards.append(-min(1.0, RISK_REWARD_CFG.omission_penalty))
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=False,
        )
        covered_gt = {m["gt_idx"] for m in matches}
        covered_pred = {m["pred_idx"] for m in matches}

        omission_rate = 1.0 - (len(covered_gt) / max(gt_count, 1))
        hallucinated = max(pred_count - len(covered_pred), 0)
        reward = (
            1.0
            - RISK_REWARD_CFG.omission_penalty * omission_rate
            - min(1.0, hallucinated * RISK_REWARD_CFG.hallucination_unit_penalty)
        )
        if no_detection:
            reward -= RISK_REWARD_CFG.omission_penalty

        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


def anomaly_instance_f1_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """实例级异常识别奖励（略偏向召回）。"""
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)
        no_detection = _is_no_detection_response(completion)

        if not gt_dets:
            if not pred_dets and no_detection:
                rewards.append(1.0)
            elif not pred_dets:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                false_alarm = sum(1 for det in pred_dets if det.is_anomaly)
                rewards.append(-min(1.0, 0.5 * false_alarm))
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=True,
        )
        if not matches:
            gt_has_anomaly = any(det.is_anomaly for det in gt_dets)
            rewards.append(-1.0 if gt_has_anomaly else 0.0)
            continue

        matched_pred_indices = {m["pred_idx"] for m in matches}
        gt_anomaly_total = sum(1 for det in gt_dets if det.is_anomaly)
        pred_anomaly_total = 0
        tp_anomaly = 0

        for m in matches:
            pred_anomaly = pred_dets[m["pred_idx"]].is_anomaly
            gt_anomaly = gt_dets[m["gt_idx"]].is_anomaly
            if pred_anomaly:
                pred_anomaly_total += 1
            if pred_anomaly and gt_anomaly:
                tp_anomaly += 1

        unmatched_pred_anomaly = sum(
            1
            for idx, det in enumerate(pred_dets)
            if idx not in matched_pred_indices and det.is_anomaly
        )
        fp_anomaly = max(pred_anomaly_total - tp_anomaly, 0) + unmatched_pred_anomaly
        fn_anomaly = max(gt_anomaly_total - tp_anomaly, 0)

        if gt_anomaly_total == 0:
            rewards.append(1.0 if fp_anomaly == 0 else max(-1.0, 1.0 - 0.5 * fp_anomaly))
            continue

        if no_detection and len(pred_dets) == 0:
            rewards.append(-1.0)
            continue

        precision = tp_anomaly / (tp_anomaly + fp_anomaly) if (tp_anomaly + fp_anomaly) > 0 else 0.0
        recall = tp_anomaly / (tp_anomaly + fn_anomaly) if (tp_anomaly + fn_anomaly) > 0 else 0.0
        reward = 0.4 * precision + 0.6 * recall
        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards


def _extract_boxes(text: str) -> List[List[int]]:
    """从文本中提取边界框，支持整数/浮点数/负数坐标。"""
    payload = _extract_json_payload(text)
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        return [det.bbox for det in _extract_detections(text)]

    boxes = []
    for match in re.finditer(BOX_PATTERN, text):
        try:
            box = [int(float(match.group(i))) for i in range(1, 5)]
            boxes.append(box)
        except (ValueError, TypeError):
            continue
    return boxes


def _extract_categories(text: str) -> List[str]:
    """从编号条目中提取类别文本。"""
    payload = _extract_json_payload(text)
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        return [det.category for det in _extract_detections(text)]

    categories = []
    pattern = r"\d+\.\s*([^\n]+)"
    for match in re.finditer(pattern, text):
        cat = match.group(1).strip()
        if "状态" in cat:
            cat = cat.split("状态")[0].strip()
        if cat:
            categories.append(cat)
    return categories


def _extract_statuses(text: str) -> List[str]:
    """提取“状态: xxx”字段。"""
    payload = _extract_json_payload(text)
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        return [det.status for det in _extract_detections(text)]

    statuses = []
    pattern = r"状态[：:]\s*([^\n]+)"
    for match in re.finditer(pattern, text):
        statuses.append(match.group(1).strip())
    return statuses


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个边界框的 IoU。"""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _normalize_category(category: str) -> str:
    """类别文本标准化：去空白并转小写。"""
    return _normalize_category_alias(category)


def _is_anomaly_status(status: str) -> bool:
    """通过关键词判断是否为异常状态。"""
    normalized = status.strip().lower()
    if normalized in NON_ANOMALY_STATES:
        return False
    if any(kw in status for kw in ANOMALY_KEYWORDS):
        return True
    return normalized not in {"", "normal"}


def _category_match(pred_category: str, gt_category: str) -> bool:
    """类别模糊匹配（完全相等或包含关系）。"""
    pred = _normalize_category(pred_category)
    gt = _normalize_category(gt_category)
    if not pred or not gt:
        return False
    if pred == "unknown" or gt == "unknown":
        return False
    return pred == gt or pred in gt or gt in pred


def _extract_detections(text: str) -> List[ParsedDetection]:
    """从输出文本解析结构化检测项（类别/状态/框）。"""
    payload = _extract_json_payload(text)
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        detections = []
        for det in payload["detections"]:
            if not isinstance(det, dict):
                continue
            parsed = _build_detection_from_json(det)
            if parsed is not None:
                detections.append(parsed)
        return detections

    detections: List[ParsedDetection] = []
    items = re.split(r"(?=\d+\.\s+)", text)

    for item in items:
        if not item.strip():
            continue

        cat_match = re.match(r"(\d+)\.\s*([^\n]+)", item)
        if not cat_match:
            continue

        category = cat_match.group(2).strip()
        status_match = re.search(r"状态[：:]\s*([^\n]+)", item)
        status = status_match.group(1).strip() if status_match else "正常"

        box_match = re.search(BOX_PATTERN, item)
        if not box_match:
            continue

        try:
            bbox = [int(float(box_match.group(i))) for i in range(1, 5)]
        except (ValueError, TypeError):
            continue

        detections.append(
            ParsedDetection(
                category=category,
                status=status,
                is_anomaly=_is_anomaly_status(status),
                bbox=bbox,
            )
        )

    if not detections:
        for box in _extract_boxes(text):
            detections.append(
                ParsedDetection(
                    category="unknown",
                    status="unknown",
                    is_anomaly=False,
                    bbox=box,
                )
            )

    return detections


def _match_detections(
    pred_dets: List[ParsedDetection],
    gt_dets: List[ParsedDetection],
    iou_threshold: float,
    require_category: bool = True,
) -> List[Dict[str, Any]]:
    """基于 IoU 的贪心一对一匹配。"""
    candidates: List[Tuple[float, int, int, bool]] = []
    for pred_idx, pred_det in enumerate(pred_dets):
        for gt_idx, gt_det in enumerate(gt_dets):
            iou = _compute_iou(pred_det.bbox, gt_det.bbox)
            if iou < iou_threshold:
                continue
            category_ok = _category_match(pred_det.category, gt_det.category)
            if require_category and not category_ok:
                continue
            score = iou + (0.01 if category_ok else 0.0)
            candidates.append((score, pred_idx, gt_idx, category_ok))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred = set()
    used_gt = set()
    matches: List[Dict[str, Any]] = []

    for _, pred_idx, gt_idx, category_ok in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        iou = _compute_iou(pred_dets[pred_idx].bbox, gt_dets[gt_idx].bbox)
        matches.append(
            {
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "iou": iou,
                "category_ok": category_ok,
            }
        )

    return matches


def _safe_f1(precision: float, recall: float) -> float:
    """安全版 F1 计算，分母为 0 时返回 0。"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    """安全版 F-beta 计算。beta>1 偏向 recall，beta<1 偏向 precision。"""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def recall_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """直接召回率奖励：匹配到的 GT 占比，强迫模型不漏检。"""
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)

        if gt_count == 0:
            # 空场景：有预测框 = 幻觉，惩罚
            if pred_count == 0 and _is_no_detection_response(completion):
                rewards.append(1.0)
            elif pred_count == 0:
                rewards.append(max(0.0, 1.0 - RISK_REWARD_CFG.no_detection_missing_penalty))
            else:
                penalty = min(1.0, pred_count * RISK_REWARD_CFG.hallucination_unit_penalty)
                rewards.append(-penalty)
            continue

        if pred_count == 0:
            # GT 有目标但模型未输出任何框：严重漏检
            rewards.append(-1.0)
            continue

        matches = _match_detections(
            pred_dets,
            gt_dets,
            iou_threshold=RISK_REWARD_CFG.match_iou_threshold,
            require_category=False,
        )
        recall = len(matches) / gt_count
        rewards.append(recall)

    return rewards


def completeness_reward(
    completions: List[str],
    assistant: Optional[List[str]] = None,
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> List[float]:
    """响应完整性奖励：惩罚 GT 有多个目标但模型只输出极少框的偷懒行为。"""
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        gt_response = _get_gt_response_text(idx, assistant)
        if not _is_format_valid(completion, gt_response):
            rewards.append(0.0)
            continue

        pred_dets = _extract_detections(completion)
        gt_dets = _get_gt_detections(idx, assistant, ground_truth)
        pred_count = len(pred_dets)
        gt_count = len(gt_dets)

        if gt_count == 0:
            # 空场景
            if pred_count == 0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            continue

        if pred_count == 0:
            rewards.append(-1.0)
            continue

        # 完整性 = min(pred/gt, 1.0)，不奖励过检
        coverage = min(pred_count / gt_count, 1.0)

        # 额外检查：输出 token 数过短时施加惩罚
        # 正常检测每个框约需 30-50 tokens，极短输出说明模型在偷懒
        completion_len = len(completion)
        expected_min_len = gt_count * 20  # 每个 GT 框至少 20 字符
        if completion_len < expected_min_len and pred_count < gt_count:
            length_penalty = max(0.0, completion_len / expected_min_len)
            coverage = coverage * length_penalty

        rewards.append(max(-1.0, min(1.0, coverage * 2.0 - 1.0)))

    return rewards
