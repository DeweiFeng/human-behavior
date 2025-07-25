# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    if response == ground_truth:
        return 1.0
    else:
        return 0.0

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                # "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "overall": accuracy_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores



# Compute score function should have the following:

# batch_inputs: List of dictionaries containing:
#             - response: The model's prediction string
#             - response_length: Length of the response
#             - ground_truth: The ground truth string
#             - segmentation_mask: Ground truth segmentation mask tensor (optional)
#             - bbox: Ground truth bounding box (optional)

# Returns: List of score dictionaries

# each score dictionary should contain:
# {
#             "overall": 0.5 * standard_score + 0.3 * iou_score + 0.1 * format_score + 0.1 * length_score,
#             "standard_score": standard_score,
#             "iou_score": iou_score,
#             "format_score": format_score,
#             "length_score": length_score,
#         }