"""Tests for layered config loading."""

from pathlib import Path

import pytest

from src.utils.config import apply_override, deep_merge, load_config, save_config


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_is_recursive_and_does_not_mutate() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 10}})
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_later_files_win_and_overrides_win_over_files(tmp_path: Path) -> None:
    first = write(tmp_path / "a.yaml", "train:\n  epochs: 10\n  lr: 0.01\n")
    second = write(tmp_path / "b.yaml", "train:\n  epochs: 20\n")
    cfg = load_config([first, second], overrides=["train.lr=0.001", "model.name=dynunet"])
    assert cfg == {"train": {"epochs": 20, "lr": 0.001}, "model": {"name": "dynunet"}}


def test_override_values_are_parsed_as_yaml() -> None:
    cfg: dict = {}
    apply_override(cfg, "roi=[128, 128, 64]")
    apply_override(cfg, "amp=true")
    assert cfg == {"roi": [128, 128, 64], "amp": True}


def test_bad_override_raises() -> None:
    with pytest.raises(ValueError):
        apply_override({}, "no_equals_sign")
    with pytest.raises(ValueError):
        apply_override({"a": 1}, "a.b=2")


def test_env_vars_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULS_TEST_USER", "alice")
    path = write(tmp_path / "m.yaml", "paths:\n  work_dir: /cluster/work/${ULS_TEST_USER}/x\n")
    assert load_config([path])["paths"]["work_dir"] == "/cluster/work/alice/x"


def test_save_and_reload_roundtrip(tmp_path: Path) -> None:
    cfg = {"seed": 42, "model": {"name": "dynunet", "roi": [128, 128, 64]}}
    save_config(cfg, tmp_path / "out" / "config.yaml")
    assert load_config([tmp_path / "out" / "config.yaml"]) == cfg
