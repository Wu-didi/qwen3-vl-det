import json

from scripts.training.rft.rewarding import (
    count_alignment_reward,
    format_reward,
    recall_reward,
    risk_control_reward,
    set_f1_reward,
)


def _json_detection(device_type="backpack_box", state="abnormal", bbox=None):
    if bbox is None:
        bbox = [100, 100, 200, 200]
    return json.dumps(
        {
            "detections": [
                {
                    "device_type": device_type,
                    "sub_type": None,
                    "state": state,
                    "bbox_1000": bbox,
                }
            ]
        },
        ensure_ascii=False,
    )


def test_risk_aware_rewards_accept_json_prediction_and_reference():
    completion = _json_detection()
    reference = _json_detection()

    assert format_reward([completion], assistant=[reference]) == [1.0]
    assert set_f1_reward([completion], assistant=[reference]) == [1.0]
    assert recall_reward([completion], assistant=[reference]) == [1.0]
    assert risk_control_reward([completion], assistant=[reference]) == [1.0]


def test_json_empty_detection_is_treated_as_no_object_response():
    completion = json.dumps({"detections": []}, ensure_ascii=False)
    reference = json.dumps({"detections": []}, ensure_ascii=False)

    assert format_reward([completion], assistant=[reference]) == [1.0]
    assert count_alignment_reward([completion], assistant=[reference]) == [1.0]
    assert risk_control_reward([completion], assistant=[reference]) == [1.0]


def test_legacy_prediction_can_match_structured_ground_truth():
    completion = (
        "1. 背包箱\n"
        "   - 状态：异常\n"
        "   - 位置：<box>(100,100),(200,200)</box>"
    )
    ground_truth = [
        {
            "detections": [
                {
                    "device_type": "backpack_box",
                    "sub_type": None,
                    "state": "abnormal",
                    "bbox_1000": [100, 100, 200, 200],
                }
            ]
        }
    ]

    assert format_reward([completion], ground_truth=ground_truth) == [1.0]
    assert set_f1_reward([completion], ground_truth=ground_truth) == [1.0]
    assert recall_reward([completion], ground_truth=ground_truth) == [1.0]
