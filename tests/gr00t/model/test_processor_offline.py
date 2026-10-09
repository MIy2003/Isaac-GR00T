"""Offline tokenizer loading must not trigger Hub metadata lookups."""

from unittest.mock import Mock

from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as module
import pytest


@pytest.mark.parametrize("offline", [True, False])
def test_processor_resolves_cached_directory(monkeypatch, tmp_path, offline):
    cached = Mock(return_value=str(tmp_path / "config.json"))
    processor = Mock()
    monkeypatch.setattr(module, "cached_file", cached)
    monkeypatch.setattr(module, "Qwen3VLProcessor", processor)
    monkeypatch.setattr(module, "is_offline_mode", lambda: offline)
    kwargs = {"revision": "pinned", "subfolder": "processor", "local_files_only": not offline}
    original = dict(kwargs)

    module.build_processor("nvidia/Cosmos-Reason2-2B", kwargs)

    cached.assert_called_once_with(
        "nvidia/Cosmos-Reason2-2B",
        "config.json",
        local_files_only=True,
        revision="pinned",
        subfolder="processor",
    )
    processor.from_pretrained.assert_called_once_with(
        str(tmp_path),
        revision="pinned",
        local_files_only=True,
    )
    assert kwargs == original


def test_processor_preserves_local_path(monkeypatch, tmp_path):
    cached = Mock(side_effect=AssertionError("Local paths must not query the Hub"))
    processor = Mock()
    monkeypatch.setattr(module, "cached_file", cached)
    monkeypatch.setattr(module, "Qwen3VLProcessor", processor)
    monkeypatch.setattr(module, "is_offline_mode", lambda: True)
    module.build_processor(str(tmp_path), {"local_files_only": True})
    processor.from_pretrained.assert_called_once_with(str(tmp_path), local_files_only=True)
