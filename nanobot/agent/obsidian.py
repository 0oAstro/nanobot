"""Helpers for prompt notes stored in an Obsidian vault."""

from __future__ import annotations

import re
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
    RETRIEVAL_FOLDERS = ("Preferences", "People", "Projects", "Library")
    _LINK_RE = re.compile(r"\[\[([^\]#|]+)")
    _TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")

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

    def retrieve_relevant_sections(
        self,
        query: str,
        *,
        max_notes: int = 4,
        max_total_chars: int = 6000,
        max_note_chars: int = 1600,
    ) -> list[str]:
        """Retrieve a few relevant note excerpts, with one-hop graph expansion."""
        terms = self._query_terms(query)
        if not terms:
            return []

        notes = self._load_retrieval_notes()
        if not notes:
            return []

        scored: list[tuple[int, str]] = []
        for rel, content in notes.items():
            score = self._score_note(rel, content, terms)
            if score > 0:
                scored.append((score, rel))
        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[str] = []
        chosen_set: set[str] = set()
        total_chars = 0

        def _try_add(rel: str) -> None:
            nonlocal total_chars
            if rel in chosen_set or len(chosen) >= max_notes:
                return
            content = notes.get(rel, "")
            excerpt = self._excerpt_for_query(content, terms, max_chars=max_note_chars)
            if not excerpt:
                return
            block = f'<note path="{rel}">\n{excerpt}\n</note>'
            if total_chars and total_chars + len(block) > max_total_chars:
                return
            chosen.append(block)
            chosen_set.add(rel)
            total_chars += len(block)

        primary = [rel for _, rel in scored[:max_notes]]
        for rel in primary:
            _try_add(rel)

        title_map = self._build_title_map(notes.keys())
        neighbors: list[tuple[int, str]] = []
        for rel in primary:
            content = notes[rel]
            links = self._linked_neighbors(content, title_map)
            backlinks = self._backlink_neighbors(rel, notes, title_map)
            for neighbor in links | backlinks:
                if neighbor in chosen_set:
                    continue
                bonus = self._score_note(neighbor, notes.get(neighbor, ""), terms)
                if bonus > 0 or neighbor in links:
                    neighbors.append((bonus + 1, neighbor))

        neighbors.sort(key=lambda item: (-item[0], item[1]))
        for _, rel in neighbors:
            _try_add(rel)

        return chosen

    def _load_retrieval_notes(self) -> dict[str, str]:
        notes: dict[str, str] = {}
        for folder in self.RETRIEVAL_FOLDERS:
            root = self.vault_path / folder
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.md")):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(self.vault_path))
                notes[rel] = path.read_text(encoding="utf-8").strip()
        return notes

    def _query_terms(self, query: str) -> list[str]:
        stop = {
            "a", "an", "and", "are", "at", "be", "for", "from", "how", "i", "in",
            "is", "it", "me", "my", "of", "on", "or", "the", "to", "us", "user",
            "was", "what", "which", "who", "with", "you", "your",
        }
        lowered = query.lower()
        terms = [tok for tok in self._TOKEN_RE.findall(lowered) if len(tok) > 1 and tok not in stop]
        return list(dict.fromkeys(terms))

    def _score_note(self, rel: str, content: str, terms: list[str]) -> int:
        haystack = f"{rel.lower()}\n{content.lower()}"
        title = Path(rel).stem.lower()
        score = 0
        for term in terms:
            if term in title:
                score += 6
            count = haystack.count(term)
            if count:
                score += min(4, count)
        return score

    def _excerpt_for_query(self, content: str, terms: list[str], *, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content
        lowered = content.lower()
        positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
        if not positions:
            return content[:max_chars].rstrip() + "\n..."
        start = max(0, min(positions) - max_chars // 5)
        end = min(len(content), start + max_chars)
        excerpt = content[start:end].strip()
        if start > 0:
            excerpt = "...\n" + excerpt
        if end < len(content):
            excerpt += "\n..."
        return excerpt

    def _build_title_map(self, rel_paths: list[str] | set[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for rel in rel_paths:
            stem = Path(rel).stem
            mapping.setdefault(stem, rel)
        return mapping

    def _linked_neighbors(self, content: str, title_map: dict[str, str]) -> set[str]:
        out: set[str] = set()
        for raw in self._LINK_RE.findall(content):
            rel = title_map.get(raw.strip())
            if rel:
                out.add(rel)
        return out

    def _backlink_neighbors(
        self,
        rel: str,
        notes: dict[str, str],
        title_map: dict[str, str],
    ) -> set[str]:
        stem = Path(rel).stem
        out: set[str] = set()
        needle = f"[[{stem}]]"
        for candidate, content in notes.items():
            if candidate != rel and needle in content:
                out.add(candidate)
        return out
