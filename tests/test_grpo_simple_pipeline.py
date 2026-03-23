import json

from scripts.data.convert_sft_to_grpo_aligned import convert_sample
from scripts.training.rft.grpo_reward_functions import (
    simple_category_recall_reward,
    simple_detection_reward,
    simple_format_reward,
    simple_state_reward,
)


def test_simple_rewards_do_not_reward_empty_output_on_positive_sample():
    completion = json.dumps({"detections": []}, ensure_ascii=False)
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

    assert simple_format_reward([completion], ground_truth=ground_truth) == [0.0]
    assert simple_detection_reward([completion], ground_truth=ground_truth) == [0.0]
    assert simple_state_reward([completion], ground_truth=ground_truth) == [0.0]
    assert simple_category_recall_reward([completion], ground_truth=ground_truth) == [0.0]


def test_simple_category_recall_penalizes_missing_hard_class():
    completion = json.dumps(
        {
            "detections": [
                {
                    "device_type": "traffic_signal",
                    "sub_type": "vehicle_signal",
                    "state": "normal",
                    "bbox_1000": [10, 10, 30, 30],
                }
            ]
        },
        ensure_ascii=False,
    )
    ground_truth = [
        {
            "detections": [
                {
                    "device_type": "traffic_signal",
                    "sub_type": "vehicle_signal",
                    "state": "normal",
                    "bbox_1000": [10, 10, 30, 30],
                },
                {
                    "device_type": "backpack_box",
                    "sub_type": None,
                    "state": "abnormal",
                    "bbox_1000": [100, 100, 220, 220],
                },
            ]
        }
    ]

    assert simple_category_recall_reward([completion], ground_truth=ground_truth) == [0.5]
    assert simple_detection_reward([completion], ground_truth=ground_truth)[0] < 1.0


def test_convert_sample_keeps_sft_prompt_and_structured_ground_truth():
    sft_sample = {
        "image": "/tmp/example.jpg",
        "conversations": [
            {"from": "user", "value": "<image>\nDetect devices and output JSON only."},
            {
                "from": "assistant",
                "value": json.dumps(
                    {
                        "detections": [
                            {
                                "device_type": "backpack_box",
                                "sub_type": None,
                                "state": "abnormal",
                                "bbox_1000": [10, 20, 22, 32],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    converted = convert_sample(sft_sample)

    assert converted["image"] == "/tmp/example.jpg"
    assert converted["prompt"] == "Detect devices and output JSON only."
    assert converted["ground_truth"]["detections"][0]["device_type"] == "backpack_box"
    assert converted["difficulty"]["num_targets"] == 1
    assert converted["difficulty"]["num_abnormal"] == 1
    assert converted["difficulty"]["has_tiny_object"] is True
