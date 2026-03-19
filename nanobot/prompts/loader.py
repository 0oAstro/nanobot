"""Prompt asset loader."""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).with_suffix("").parent


def load_prompt(name: str, **values: object) -> str:
    template = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return template.format(**values)
