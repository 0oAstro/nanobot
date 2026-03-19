"""Tools for tracked subagent orchestration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool
from nanobot.runs import WaitRequest

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.runs import RunManager


class SpawnSubagentTool(Tool):
    """Spawn tracked subagents and return a stable run id."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._parent_run_id: str | None = None

    def set_context(self, channel: str, chat_id: str, parent_run_id: str | None = None) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._parent_run_id = parent_run_id

    @property
    def name(self) -> str:
        return "spawn_subagent"

    @property
    def description(self) -> str:
        return "Spawn a tracked subagent and return its run id."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task for the subagent to complete"},
                "label": {"type": "string", "description": "Optional short label for the subagent"},
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        payload = await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            parent_run_id=self._parent_run_id,
        )
        return json.dumps(payload, ensure_ascii=False)


class WaitForSubagentsTool(Tool):
    """Suspend parent processing until child runs are ready."""

    def __init__(self, runs: "RunManager"):
        self._runs = runs
        self._parent_run_id: str | None = None

    def set_context(self, parent_run_id: str | None = None) -> None:
        self._parent_run_id = parent_run_id

    @property
    def name(self) -> str:
        return "wait_for_subagents"

    @property
    def description(self) -> str:
        return "Suspend until tracked subagents complete, then resume from their handoffs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "mode": {
                    "type": "string",
                    "enum": ["all_completed_or_any_actionable"],
                },
            },
        }

    async def execute(
        self,
        run_ids: list[str] | None = None,
        mode: str = "all_completed_or_any_actionable",
        **kwargs: Any,
    ) -> str:
        if not self._parent_run_id:
            return "Error: wait_for_subagents requires an active parent run"
        targets = run_ids or []
        completed = self._runs.get_completed_events(self._parent_run_id, targets or None)
        if completed:
            return json.dumps({"ready": True, "events": completed}, ensure_ascii=False)
        self._runs.record_wait_request(
            self._parent_run_id,
            WaitRequest(run_ids=targets, mode=mode),
        )
        return json.dumps({"ready": False, "waiting_on": targets, "mode": mode}, ensure_ascii=False)


class GetSubagentStatusTool(Tool):
    """Inspect tracked child runs."""

    def __init__(self, runs: "RunManager"):
        self._runs = runs
        self._parent_run_id: str | None = None

    def set_context(self, parent_run_id: str | None = None) -> None:
        self._parent_run_id = parent_run_id

    @property
    def name(self) -> str:
        return "get_subagent_status"

    @property
    def description(self) -> str:
        return "Inspect the status of tracked child runs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "run_ids": {"type": "array", "items": {"type": "string"}},
            },
        }

    async def execute(self, run_ids: list[str] | None = None, **kwargs: Any) -> str:
        if not self._parent_run_id:
            return "Error: get_subagent_status requires an active parent run"
        children = self._runs.list_children(self._parent_run_id)
        if run_ids:
            wanted = set(run_ids)
            children = [child for child in children if child.run_id in wanted]
        payload = [
            {
                "run_id": child.run_id,
                "label": child.label,
                "status": child.status,
                "updated_at": child.updated_at,
                "result_handoff": child.result_handoff,
            }
            for child in children
        ]
        return json.dumps(payload, ensure_ascii=False)


class CancelSubagentTool(Tool):
    """Cancel a tracked child run."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "cancel_subagent"

    @property
    def description(self) -> str:
        return "Cancel a tracked child run by id."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
        }

    async def execute(self, run_id: str, **kwargs: Any) -> str:
        cancelled = await self._manager.cancel_run(run_id)
        return json.dumps({"run_id": run_id, "cancelled": cancelled}, ensure_ascii=False)
