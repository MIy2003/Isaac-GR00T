"""Two-step capacity check; all non-duration settings come from the base run."""
import argparse
import json
from pathlib import Path
import torch
from examples.CHIP.train_ta_force import prepare_config
from gr00t.experiment.experiment import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    config, diff = prepare_config(args.base_config, args.output_dir)
    assert config.training.global_batch_size == 160
    assert config.training.gradient_accumulation_steps == 1
    assert config.model.rtc_training_max_delay == 6
    config.training.max_steps = 2
    config.training.logging_steps = 1
    run(config)
    report = dict(gpu=torch.cuda.get_device_name(),
                  peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30,
                  global_batch=160, accumulation=1, rtc_max_delay=6, base_config_changes=diff)
    Path(args.output_dir, 'capacity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
