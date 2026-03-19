"""Structured handoff tool for subagents."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class ReturnHandoffTool(Tool):
    """Capture a single structured handoff from a subagent."""

    def __init__(self) -> None:
        self._handoff: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "return_handoff"

    @property
    def description(self) -> str:
        return "Return the final structured handoff for this subagent task and stop."

    @property
    def parameters(self) -> dict[str, Any]:
        issue_schema = {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                "description": {"type": "string"},
                "suggested_fix": {"type": "string"},
            },
            "required": ["severity", "description"],
        }
        evidence_schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "value": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["type", "value", "note"],
        }
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "partial", "failure"]},
                "return_to_orchestrator": {"type": "boolean"},
                "summary": {"type": "string"},
                "what_was_done": {"type": "string"},
                "what_remains": {"type": "string"},
                "evidence": {"type": "array", "items": evidence_schema},
                "discovered_issues": {"type": "array", "items": issue_schema},
                "next_action": {"type": "string"},
            },
            "required": [
                "status",
                "return_to_orchestrator",
                "summary",
                "what_was_done",
                "what_remains",
                "evidence",
                "discovered_issues",
                "next_action",
            ],
        }

    async def execute(self, **kwargs: Any) -> str:
        self._handoff = dict(kwargs)
        return "Structured handoff recorded. Stop now."

    @property
    def handoff(self) -> dict[str, Any] | None:
        return self._handoff
