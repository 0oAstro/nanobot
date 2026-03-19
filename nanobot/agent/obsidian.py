"""Helpers for prompt notes stored in an Obsidian vault."""

from __future__ import annotations

from pathlib import Path

from nanobot.config.paths import get_obsidian_vault_path
from nanobot.utils.helpers import ensure_dir


class ObsidianVault:
    """Read and initialize prompt-related notes in an Obsidian vault."""

    DEFAULT_FOLDERS = (
        "Preferences",
        "People",
        "Projects",
        "Library",
        "Checkpoints",
        "Scratch",
    )

    def __init__(self, workspace: Path, configured_path: str | None = None):
        self.workspace = workspace
        self.vault_path = get_obsidian_vault_path(workspace, configured_path)

    @property
    def user_file(self) -> Path:
        return self.vault_path / "USER.md"

    @property
    def soul_file(self) -> Path:
        return self.vault_path / "SOUL.md"

    def init_prompt_notes(self, user_template: str, soul_template: str) -> list[Path]:
        """Create the Obsidian vault plus prompt-note files if missing."""
        created: list[Path] = []
        ensure_dir(self.vault_path)
        ensure_dir(self.vault_path / ".obsidian")
        for folder in self.DEFAULT_FOLDERS:
            folder_path = self.vault_path / folder
            if folder_path.exists():
                continue
            ensure_dir(folder_path)
            created.append(folder_path)

        for path, content in (
            (self.user_file, user_template),
            (self.soul_file, soul_template),
        ):
            if path.exists():
                continue
            path.write_text(content, encoding="utf-8")
            created.append(path)

        return created

    def read_prompt_sections(self) -> list[str]:
        """Return the prompt-note contents to embed into the system prompt."""
        sections: list[str] = []
        for tag, path in (("soul_note", self.soul_file), ("user_note", self.user_file)):
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8").strip()
            if content:
                sections.append(f"<{tag}>\n{content}\n</{tag}>")
        return sections
