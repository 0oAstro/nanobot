# Repository Guidelines

## Codebase Map
The repository has a Python runtime plus a small TypeScript bridge.

```text
.
├── AGENTS.md
├── pyproject.toml
├── uv.lock
├── docs/
│   └── channel/plugin docs and operational notes
├── bridge/
│   ├── package.json
│   └── src/
│       ├── server.ts
│       ├── whatsapp.ts
│       └── index.ts
├── nanobot/
│   ├── __main__.py
│   ├── runtime.py
│   ├── agent/
│   │   ├── loop.py
│   │   ├── subagent.py
│   │   ├── context.py
│   │   ├── memory.py
│   │   └── tools/
│   ├── bus/
│   ├── channels/
│   ├── cli/
│   ├── config/
│   ├── cron/
│   ├── heartbeat/
│   ├── mission/
│   │   ├── manager.py
│   │   └── prompts.py
│   ├── prompts/
│   ├── providers/
│   ├── runs/
│   │   └── manager.py
│   ├── security/
│   ├── session/
│   │   └── manager.py
│   ├── skills/
│   ├── templates/
│   └── utils/
└── tests/
    └── pytest suite for providers, channels, sessions, missions, and tools
```

## Project Structure & Module Organization
Core runtime code lives in `nanobot/`. The main execution path is in `nanobot/agent/`, while transport adapters live in `nanobot/channels/`, provider integrations in `nanobot/providers/`, durable parent/child orchestration in `nanobot/runs/`, and mission orchestration in `nanobot/mission/`. Session persistence lives in `nanobot/session/`, config loading/path resolution in `nanobot/config/`, and reusable prompts/templates in `nanobot/prompts/` and `nanobot/templates/`.

Tests live in top-level `tests/` and usually mirror the feature or subsystem name. The `bridge/` directory is a separate TypeScript service surface; do not assume Python conventions apply there. `nanobot/nanobot/` currently exists as an empty artifact path and is not the package root.

## Runtime Pipeline
```text
CLI / channel input
  -> nanobot/cli/commands.py bootstraps config, workspace, provider, bus, cron, channels
  -> nanobot/agent/loop.py receives inbound messages and handles slash commands
  -> nanobot/session/manager.py loads recent JSONL session history
  -> nanobot/agent/context.py builds the model prompt from bootstrap docs, skills, history, MEMORY.md
  -> nanobot/runs/manager.py creates or resumes tracked parent runs under workspace/runs/
  -> nanobot/providers/*.py executes the model call
  -> nanobot/agent/tools/* executes tool calls
  -> nanobot/agent/subagent.py runs tracked child workers when spawned
  -> nanobot/mission/manager.py plans, approves, dispatches, and reconciles mission work on top of runs/
  -> nanobot/agent/memory.py consolidates older turns into workspace/memory/MEMORY.md and HISTORY.md
  -> outbound response is published back to CLI or channel
```

## Mission Runtime Notes
- `/mission` is the only user-facing mission entrypoint.
- Mission planning writes durable artifacts under the global missions directory, including `mission.json`, `mission.md`, `state.json`, `features.json`, `validation-contract.md`, `validation-state.json`, `handoffs/`, `handoffs.jsonl`, and `progress_log.jsonl`.
- Mission execution is layered on `nanobot/runs/manager.py`, not a separate runtime.
- Canonical mission files are orchestrator-owned. Mission workers must not mutate `features.json`, `state.json`, `validation-contract.md`, `validation-state.json`, or mission-local `AGENTS.md`.
- Use the tracked orchestration tools only: `spawn_subagent`, `wait_for_subagents`, `get_subagent_status`, and `cancel_subagent`.
- Do not add backward-compat aliases for removed orchestration surfaces unless explicitly requested.

## Build, Test, and Development Commands
Use `uv` for Python setup and execution.

- `uv sync --extra dev` installs runtime and dev dependencies.
- `uv run pytest -q` runs the full Python test suite.
- `uv run pytest tests/test_mission_manager.py -q` runs the focused mission suite.
- `uv run pytest tests/test_consolidate_offset.py -q` runs the session-history regression slice.
- `uv run python -m nanobot --help` or `uv run nanobot --help` checks the CLI entrypoint.

If you need TypeScript bridge work, use the `bridge/` package tooling from that directory.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation and type hints on public interfaces. Follow existing local patterns instead of introducing new abstractions without reason. Modules and functions use `snake_case`, classes use `PascalCase`, and tests use `test_*.py`. Keep line length aligned with `pyproject.toml`.

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio` enabled via `asyncio_mode = "auto"`. Add or update tests for behavior changes, especially around CLI commands, channel integrations, session handling, mission orchestration, and tool execution. Prefer narrowly scoped tests with descriptive names. Run focused tests during iteration, then run `uv run pytest -q` before closing the task.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit style such as `feat: ...`, `fix(providers): ...`, and `test(custom): ...`. Keep commit subjects imperative and specific. PRs should summarize the behavioral or architectural change, note test coverage, and link the relevant issue or discussion.

## Security & Configuration Tips
Do not commit secrets or runtime instance data. Live runtime files belong under `~/.nanobot/`, not this repository. When editing config or runtime path logic, verify both the code and the deployed layout because runtime directories are derived from the active config location.
