import os
from pathlib import Path

from nanobot.config.loader import load_config


def test_load_config_applies_runtime_environment(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        """
        {
          "environment": {
            "FOO_TOKEN": "abc123",
            "BAR_PATH": "~/demo/$HOME_SUFFIX"
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("FOO_TOKEN", raising=False)
    monkeypatch.delenv("BAR_PATH", raising=False)
    monkeypatch.setenv("HOME_SUFFIX", "nested")

    load_config(config_file)

    assert os.environ["FOO_TOKEN"] == "abc123"
    assert os.environ["BAR_PATH"] == str(Path.home() / "demo" / "nested")
