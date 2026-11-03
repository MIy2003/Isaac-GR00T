"""Train with an exact base experiment configuration, adding only TA Force."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from gr00t.configs.base_config import Config, _build_safe_tree
from gr00t.model.modules.ta_force import TAForceConfig
from gr00t.experiment.experiment import run


def prepare_config(base_config, output_dir):
    config = Config.from_pretrained(Path(base_config))
    # Saved YAML contains plain dicts; restore typed modality configs without changing values.
    from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
    for modalities in config.data.modality_configs.values():
        for key, value in modalities.items():
            if isinstance(value, dict):
                value = dict(value)
                if value.get('action_configs') is not None:
                    value['action_configs'] = [ActionConfig(
                        rep=ActionRepresentation(a['rep']), type=ActionType(a['type']),
                        format=ActionFormat(a['format']), state_key=a.get('state_key'))
                        for a in value['action_configs']]
                modalities[key] = ModalityConfig(**value)
    before = _build_safe_tree(asdict(config))
    if config.model.force_mot is not None or config.model.ta_force is not None:
        raise ValueError('Expected a plain base policy experiment')
    config.model.ta_force = asdict(TAForceConfig())
    config.training.output_dir = str(output_dir)
    # Keep the original experiment name, optimizer, sampler, processor and initialization.
    after = _build_safe_tree(asdict(config))
    diff = {}
    for section in ('model', 'data', 'training'):
        for key in set(before[section]) | set(after[section]):
            if before[section].get(key) != after[section].get(key):
                diff[f'{section}.{key}'] = [before[section].get(key), after[section].get(key)]
    if set(diff) != {'model.ta_force', 'training.output_dir'}:
        raise ValueError(f'Unexpected base configuration drift: {diff}')
    return config, diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir)
    config, diff = prepare_config(args.base_config, output)
    print(json.dumps(dict(base_config=args.base_config, changes=diff,
                          initialization=config.training.start_from_checkpoint), indent=2), flush=True)
    if args.audit_only:
        return
    if output.exists():
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    (output / 'base_config_comparison.json').write_text(json.dumps(diff, indent=2))
    run(config)


if __name__ == '__main__':
    main()
