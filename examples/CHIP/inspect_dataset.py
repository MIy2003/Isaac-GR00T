"""Validate CHIP samples/statistics on CPU before loading any model weights.

Run from the repo root: python -m examples.CHIP.inspect_dataset --dataset-path ...
"""

import argparse

from gr00t.configs.data.chip_native_config import CHIP_ACTION_HORIZON, chip_native_modality_config
from gr00t.data.dataset.chip_native_dataset import ChipNativeDataset
from gr00t.data.state_action.state_action_processor import StateActionProcessor
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--task-description", default="Perform the demonstrated task.")
    args = parser.parse_args()
    config = chip_native_modality_config()
    dataset = ChipNativeDataset(args.dataset_path, config, task_description=args.task_description)
    tag = dataset.embodiment_tag.value
    processor = StateActionProcessor(
        {tag: config},
        {tag: dataset.get_dataset_statistics()},
        use_percentiles=False,
        use_relative_action=False,
    )
    for index in sorted({0, len(dataset.samples) // 2, len(dataset.samples) - 1}):
        step = dataset.get_step(index)
        normalized = processor.apply_action(step.actions, tag, step.states)
        decoded = processor.unapply_action(normalized, tag, step.states)
        np.testing.assert_allclose(
            decoded["native_relative"], step.actions["native_relative"], atol=1e-5
        )
    print(f"Validated {len(dataset.episode_names)} train episodes / {len(dataset.samples)} samples")
    print(f"RGB [2,224,400,3]; native state [3,37]; relative action [{CHIP_ACTION_HORIZON},37]")
    print("Normalization round trip passed; model weights were not loaded.")


if __name__ == "__main__":
    main()
