"""Detection prompt templates for traffic equipment anomaly detection."""

from enum import Enum


class AnomalyCategory(str, Enum):
    """Categories of traffic equipment anomalies."""

    TRAFFIC_SIGN = "traffic_sign"  # 交通标志
    TRAFFIC_LIGHT = "traffic_light"  # 交通信号灯
    ROAD_FACILITY = "road_facility"  # 道路设施
    GUIDANCE_SCREEN = "guidance_screen"  # 诱导屏
    HEIGHT_LIMIT = "height_limit"  # 限高架
    CABINET = "cabinet"  # 机柜


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    # Traffic sign anomalies
    DAMAGED = "damaged"  # 损坏
    MISSING = "missing"  # 缺失
    BLOCKED = "blocked"  # 遮挡
    FADED = "faded"  # 褪色
    TILTED = "tilted"  # 倾斜

    # Traffic light anomalies
    MALFUNCTION = "malfunction"  # 故障
    BULB_BROKEN = "bulb_broken"  # 灯泡损坏
    NOT_LIT = "not_lit"  # 不亮
    COLOR_ABNORMAL = "color_abnormal"  # 颜色异常

    # Road facility anomalies
    GUARDRAIL_DAMAGED = "guardrail_damaged"  # 护栏损坏
    ROAD_DAMAGED = "road_damaged"  # 路面破损
    LINE_WORN = "line_worn"  # 标线磨损
    MANHOLE_MISSING = "manhole_missing"  # 井盖缺失

    # Guidance screen anomalies
    DISPLAY_ERROR = "display_error"  # 显示故障
    BLACK_SCREEN = "black_screen"  # 黑屏
    GARBLED = "garbled"  # 乱码

    # Height limit anomalies
    STRUCTURE_DAMAGED = "structure_damaged"  # 结构损坏
    MARKING_UNCLEAR = "marking_unclear"  # 标识不清

    # Cabinet anomalies
    DOOR_OPEN = "door_open"  # 柜门未关闭
    CABINET_DAMAGED = "cabinet_damaged"  # 机柜损坏

    # General
    OTHER = "other"  # 其他异常
    NORMAL = "normal"  # 正常


# Main detection prompt for comprehensive traffic equipment inspection
DETECTION_PROMPT = """你是一个专业的交通设备异常检测系统。请仔细分析这张图片，检测以下类型的交通设备异常：

## 检测类别

1. **交通标志 (traffic_sign)**
   - 损坏 (damaged): 标志牌破损、变形
   - 缺失 (missing): 标志牌丢失、不完整
   - 遮挡 (blocked): 被树木、广告等遮挡
   - 褪色 (faded): 颜色褪色、不清晰
   - 倾斜 (tilted): 标志牌倾斜、歪斜

2. **交通信号灯 (traffic_light)**
   - 故障 (malfunction): 信号灯不工作
   - 灯泡损坏 (bulb_broken): 灯泡破损
   - 不亮 (not_lit): 应亮未亮
   - 颜色异常 (color_abnormal): 颜色显示异常

3. **道路设施 (road_facility)**
   - 护栏损坏 (guardrail_damaged): 护栏变形、缺失
   - 路面破损 (road_damaged): 路面坑洼、裂缝
   - 标线磨损 (line_worn): 道路标线模糊、缺失
   - 井盖缺失 (manhole_missing): 井盖丢失或损坏

4. **诱导屏 (guidance_screen)**
   - 显示故障 (display_error): 显示不正常
   - 黑屏 (black_screen): 屏幕不亮
   - 乱码 (garbled): 显示内容乱码

5. **限高架 (height_limit)**
   - 结构损坏 (structure_damaged): 限高架变形、损坏
   - 标识不清 (marking_unclear): 高度标识不清晰

6. **机柜 (cabinet)**
   - 柜门未关闭 (door_open): 控制柜门敞开
   - 机柜损坏 (cabinet_damaged): 机柜外壳损坏

## 输出要求

请以JSON格式输出检测结果。如果检测到异常，按以下格式输出：

```json
{
  "has_anomaly": true,
  "detections": [
    {
      "category": "异常类别英文名",
      "anomaly_type": "异常类型英文名",
      "confidence": 0.85,
      "bbox": [x1, y1, x2, y2],
      "description": "异常的详细中文描述"
    }
  ],
  "summary": "整体检测结果的中文总结"
}
```

如果没有检测到异常：

```json
{
  "has_anomaly": false,
  "detections": [],
  "summary": "未检测到交通设备异常，设备状态正常"
}
```

## 注意事项

- bbox坐标为归一化坐标 [0-1000]，格式为 [左上角x, 左上角y, 右下角x, 右下角y]
- confidence为置信度，范围0-1
- 只报告明确可见的异常，不要猜测
- 对于每个检测到的异常，提供准确的位置和详细描述

请分析图片并输出检测结果："""


# Simplified prompt for quick detection
QUICK_DETECTION_PROMPT = """分析图片中的交通设备，检测是否存在异常（损坏、故障、缺失等）。

以JSON格式输出：
```json
{
  "has_anomaly": true/false,
  "detections": [{"category": "类别", "anomaly_type": "异常类型", "confidence": 0.0-1.0, "bbox": [x1,y1,x2,y2], "description": "描述"}],
  "summary": "总结"
}
```

请检测："""


# Category-specific prompts
CATEGORY_PROMPTS = {
    AnomalyCategory.TRAFFIC_SIGN: """检测图片中的交通标志是否存在以下异常：
- 损坏：标志牌破损、变形
- 缺失：标志牌丢失、不完整
- 遮挡：被树木、广告等遮挡
- 褪色：颜色褪色、不清晰
- 倾斜：标志牌倾斜、歪斜

以JSON格式输出检测结果，包含 has_anomaly, detections（含category, anomaly_type, confidence, bbox, description）, summary。
bbox格式为归一化坐标 [x1,y1,x2,y2]，范围0-1000。""",
    AnomalyCategory.TRAFFIC_LIGHT: """检测图片中的交通信号灯是否存在以下异常：
- 故障：信号灯不工作
- 灯泡损坏：灯泡破损
- 不亮：应亮未亮
- 颜色异常：颜色显示异常

以JSON格式输出检测结果。""",
    AnomalyCategory.ROAD_FACILITY: """检测图片中的道路设施是否存在以下异常：
- 护栏损坏：护栏变形、缺失
- 路面破损：路面坑洼、裂缝
- 标线磨损：道路标线模糊、缺失
- 井盖缺失：井盖丢失或损坏

以JSON格式输出检测结果。""",
    AnomalyCategory.GUIDANCE_SCREEN: """检测图片中的诱导屏/显示屏是否存在以下异常：
- 显示故障：显示不正常
- 黑屏：屏幕不亮
- 乱码：显示内容乱码

以JSON格式输出检测结果。""",
    AnomalyCategory.HEIGHT_LIMIT: """检测图片中的限高架是否存在以下异常：
- 结构损坏：限高架变形、损坏
- 标识不清：高度标识不清晰

以JSON格式输出检测结果。""",
    AnomalyCategory.CABINET: """检测图片中的机柜（控制柜、配电柜等）是否存在以下异常：
- 柜门未关闭：控制柜门敞开
- 机柜损坏：机柜外壳损坏

以JSON格式输出检测结果。""",
}


def get_detection_prompt(
    categories: list[AnomalyCategory] | None = None,
    quick_mode: bool = False,
) -> str:
    """Get the appropriate detection prompt.

    Args:
        categories: Specific categories to detect. If None, detects all.
        quick_mode: Use simplified prompt for faster inference.

    Returns:
        Detection prompt string.
    """
    if quick_mode:
        return QUICK_DETECTION_PROMPT

    if categories is None or len(categories) == 0:
        return DETECTION_PROMPT

    if len(categories) == 1:
        return CATEGORY_PROMPTS.get(categories[0], DETECTION_PROMPT)

    # Multiple categories - combine prompts
    prompt_parts = ["请检测图片中以下类型的交通设备异常：\n"]
    for cat in categories:
        if cat in CATEGORY_PROMPTS:
            prompt_parts.append(f"\n{CATEGORY_PROMPTS[cat]}\n")

    prompt_parts.append(
        "\n以JSON格式输出检测结果，包含 has_anomaly, detections, summary。"
    )
    return "".join(prompt_parts)
