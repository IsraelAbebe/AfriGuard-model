# Copyright 2025 the LlamaFactory team.
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

"""Build the non-instruction ("plain") version of AfriGuard-inst.

Each example keeps only the user prompt and the text inside ``<response>...</response>``;
the safety system instruction and the ``<safety>`` / ``<category>`` / ``<response>`` tags are dropped.

Usage:
    python scripts/afriguard_plain.py
"""

import json
import os
import re

import fire
from datasets import load_dataset


RESPONSE_PATTERN = re.compile(r"<response>(.*?)</response>", re.DOTALL)
TAG_PATTERN = re.compile(r"</?(safety|category|response)>")
METADATA_COLUMNS = ["id", "source", "config", "language", "safe_unsafe", "top_category"]


def convert(example: dict) -> dict:
    match = RESPONSE_PATTERN.search(example["output"])
    if match is None:
        raise ValueError(f"No <response> tag in example {example['id']}.")

    response = match.group(1).strip()
    if TAG_PATTERN.search(response):
        raise ValueError(f"Leftover tags in response of example {example['id']}.")

    record = {"instruction": example["input"], "input": "", "output": response}
    record.update({column: example[column] for column in METADATA_COLUMNS})
    return record


def main(dataset: str = "israel/AfriGuard-inst", output_dir: str = "data", prefix: str = "afriguard_plain"):
    for split, split_dataset in load_dataset(dataset).items():
        records = [convert(example) for example in split_dataset]
        output_path = os.path.join(output_dir, f"{prefix}_{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(records)} examples to {output_path}.")


if __name__ == "__main__":
    fire.Fire(main)
