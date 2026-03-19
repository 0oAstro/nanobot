"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from importlib.resources import files as pkg_files
from pathlib import Path
import datetime as datetime_module

from nanobot.agent.context import ContextBuilder


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_bootstrap_files_are_backed_by_templates() -> None:
    template_dir = pkg_files("nanobot") / "templates"

    assert (template_dir / "AGENTS.md").is_file()
    assert (template_dir / "USER.md").is_file()
    assert (template_dir / "SOUL.md").is_file()


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be merged with the user message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    assert messages[0]["role"] == "system"
    assert "## Current Session" not in messages[0]["content"]

    # Runtime context is now merged with user message into a single message
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, str)
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in user_content
    assert "Current Time:" in user_content
    assert "Channel: cli" in user_content
    assert "Chat ID: direct" in user_content
    assert "Return exactly: OK" in user_content


def test_system_prompt_retrieves_relevant_obsidian_notes(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    vault = workspace / "obsidian-vault"
    (vault / "People").mkdir(parents=True)
    (vault / "Projects").mkdir(parents=True)
    (vault / "USER.md").write_text("# User\n", encoding="utf-8")
    (vault / "SOUL.md").write_text("# Soul\n", encoding="utf-8")
    (vault / "People" / "Tomas Brennan.md").write_text(
        "# Tomas Brennan\nLeads the migration from MySQL 8.0 to DuckDB 1.1.\n",
        encoding="utf-8",
    )
    (vault / "Projects" / "Analytics Migration.md").write_text(
        "# Analytics Migration\nWe are migrating from MySQL 8.0 to DuckDB 1.1 with [[Tomas Brennan]].\n",
        encoding="utf-8",
    )

    builder = ContextBuilder(workspace)
    prompt = builder.build_system_prompt(current_message="Who is leading the DuckDB migration?")

    assert "<retrieved_notes>" in prompt
    assert 'path="Projects/Analytics Migration.md"' in prompt
    assert 'path="People/Tomas Brennan.md"' in prompt
    assert "DuckDB 1.1" in prompt
