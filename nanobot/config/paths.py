"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

from pathlib import Path

from nanobot.config.loader import get_config_path
from nanobot.utils.helpers import ensure_dir


def get_data_dir() -> Path:
    """Return the instance-level runtime data directory."""
    return ensure_dir(get_config_path().parent)


def get_global_runtime_dir() -> Path:
    """Return the shared global runtime directory."""
    return ensure_dir(Path.home() / ".nanobot")


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory under the instance data dir."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory, optionally namespaced per channel."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """Return the cron storage directory."""
    return get_runtime_subdir("cron")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_missions_dir() -> Path:
    """Return the shared mission storage directory."""
    return ensure_dir(get_global_runtime_dir() / "missions")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path."""
    path = Path(workspace).expanduser() if workspace else Path.home() / ".nanobot" / "workspace"
    return ensure_dir(path)


def get_obsidian_vault_path(workspace: Path, configured_path: str | None = None) -> Path:
    """Resolve and ensure the Obsidian vault path."""
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = workspace / path
    else:
        path = workspace / "obsidian-vault"
    return ensure_dir(path)


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return get_global_runtime_dir() / "history" / "cli_history"


def get_bridge_install_dir() -> Path:
    """Return the shared WhatsApp bridge installation directory."""
    return get_global_runtime_dir() / "bridge"
