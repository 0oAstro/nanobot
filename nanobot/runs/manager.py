"""Persistent orchestration state for parent and child runs."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nanobot.utils.helpers import ensure_dir

RunStatus = Literal["running", "waiting", "completed", "failed", "cancelled"]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


@dataclass
class ChildHandoff:
    """Compact structured child completion payload."""

    status: str
    return_to_orchestrator: bool
    summary: str
    what_was_done: str
    what_remains: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    discovered_issues: list[dict[str, str]] = field(default_factory=list)
    next_action: str = ""
    error: str = ""

    @classmethod
    def normalize(cls, payload: Any, *, fallback_error: str = "") -> "ChildHandoff":
        data = payload if isinstance(payload, dict) else {}
        status = str(data.get("status") or ("failure" if fallback_error else "success")).lower()
        if status not in {"success", "partial", "failure"}:
            status = "failure" if fallback_error else "partial"
        issues: list[dict[str, str]] = []
        for item in _as_list(data.get("discovered_issues")):
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "medium").lower()
            if severity not in {"high", "medium", "low"}:
                severity = "medium"
            description = str(item.get("description") or "").strip()
            if not description:
                continue
            norm = {"severity": severity, "description": description}
            if item.get("suggested_fix"):
                norm["suggested_fix"] = str(item["suggested_fix"])
            issues.append(norm)
        evidence: list[dict[str, str]] = []
        for item in _as_list(data.get("evidence")):
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "type": str(item.get("type") or "observation"),
                    "value": str(item.get("value") or ""),
                    "note": str(item.get("note") or ""),
                }
            )
        summary = str(data.get("summary") or fallback_error or "Task completed.")
        what_done = str(data.get("what_was_done") or summary)
        what_remains = str(data.get("what_remains") or "")
        next_action = str(data.get("next_action") or "")
        if fallback_error and not issues:
            issues.append({"severity": "high", "description": fallback_error})
        return cls(
            status=status,
            return_to_orchestrator=bool(
                data.get("return_to_orchestrator", status != "success" or bool(issues) or bool(what_remains))
            ),
            summary=summary,
            what_was_done=what_done,
            what_remains=what_remains,
            evidence=evidence,
            discovered_issues=issues,
            next_action=next_action,
            error=fallback_error,
        )


@dataclass
class WaitRequest:
    """Parent wait intent captured by the wait tool."""

    run_ids: list[str] = field(default_factory=list)
    mode: str = "all_completed_or_any_actionable"


@dataclass
class RunRecord:
    """Persistent parent or child run state."""

    run_id: str
    session_key: str
    kind: Literal["parent", "subagent"]
    status: RunStatus
    goal: str
    parent_run_id: str | None = None
    waiting_on: list[str] = field(default_factory=list)
    queued_events: list[dict[str, Any]] = field(default_factory=list)
    wake_queued: bool = False
    pending_messages: list[dict[str, Any]] = field(default_factory=list)
    result_handoff: dict[str, Any] | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    label: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=data["run_id"],
            session_key=data["session_key"],
            kind=data["kind"],
            status=data["status"],
            goal=data.get("goal", ""),
            parent_run_id=data.get("parent_run_id"),
            waiting_on=_as_list(data.get("waiting_on")),
            queued_events=_as_list(data.get("queued_events")),
            wake_queued=bool(data.get("wake_queued", False)),
            pending_messages=_as_list(data.get("pending_messages")),
            result_handoff=data.get("result_handoff"),
            created_at=data.get("created_at", _utc_now()),
            updated_at=data.get("updated_at", _utc_now()),
            label=str(data.get("label") or ""),
        )


class RunManager:
    """Manage durable parent and child runs."""

    def __init__(self, workspace: Path):
        self.runs_dir = ensure_dir(workspace / "runs")
        self._wait_requests: dict[str, WaitRequest] = {}

    def _path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _save(self, record: RunRecord) -> RunRecord:
        record.updated_at = _utc_now()
        self._path(record.run_id).write_text(
            json.dumps(asdict(record), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return record

    def load(self, run_id: str) -> RunRecord | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        return RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def create_parent_run(self, *, session_key: str, goal: str, messages: list[dict[str, Any]]) -> RunRecord:
        run = RunRecord(
            run_id=str(uuid.uuid4())[:8],
            session_key=session_key,
            kind="parent",
            status="running",
            goal=goal,
            pending_messages=messages,
        )
        return self._save(run)

    def create_subagent_run(
        self,
        *,
        parent_run_id: str,
        session_key: str,
        goal: str,
        label: str,
    ) -> RunRecord:
        run = RunRecord(
            run_id=str(uuid.uuid4())[:8],
            session_key=session_key,
            kind="subagent",
            status="running",
            goal=goal,
            parent_run_id=parent_run_id,
            label=label,
        )
        self._save(run)
        if parent := self.load(parent_run_id):
            if run.run_id not in parent.waiting_on:
                parent.waiting_on.append(run.run_id)
            self._save(parent)
        return run

    def set_pending_messages(self, run_id: str, messages: list[dict[str, Any]]) -> RunRecord | None:
        if not (run := self.load(run_id)):
            return None
        run.pending_messages = messages
        return self._save(run)

    def set_status(self, run_id: str, status: RunStatus) -> RunRecord | None:
        if not (run := self.load(run_id)):
            return None
        run.status = status
        return self._save(run)

    def mark_waiting(self, run_id: str, messages: list[dict[str, Any]], run_ids: list[str]) -> RunRecord | None:
        if not (run := self.load(run_id)):
            return None
        run.status = "waiting"
        run.pending_messages = messages
        run.waiting_on = run_ids or run.waiting_on
        return self._save(run)

    def complete_subagent(self, run_id: str, handoff: ChildHandoff) -> tuple[RunRecord | None, RunRecord | None]:
        child = self.load(run_id)
        if not child:
            return None, None
        child.status = "completed" if handoff.status == "success" else "failed" if handoff.status == "failure" else "completed"
        child.result_handoff = asdict(handoff)
        self._save(child)
        parent = self.load(child.parent_run_id) if child.parent_run_id else None
        if parent:
            parent.queued_events.append({"run_id": child.run_id, "handoff": asdict(handoff)})
            self._save(parent)
        return child, parent

    def cancel_run(self, run_id: str) -> RunRecord | None:
        if not (run := self.load(run_id)):
            return None
        run.status = "cancelled"
        return self._save(run)

    def list_children(self, parent_run_id: str) -> list[RunRecord]:
        out: list[RunRecord] = []
        for path in self.runs_dir.glob("*.json"):
            try:
                record = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if record.parent_run_id == parent_run_id:
                out.append(record)
        return sorted(out, key=lambda item: item.created_at)

    def get_completed_events(self, parent_run_id: str, run_ids: list[str] | None = None) -> list[dict[str, Any]]:
        parent = self.load(parent_run_id)
        if not parent:
            return []
        events = parent.queued_events
        if run_ids:
            wanted = set(run_ids)
            events = [event for event in events if event.get("run_id") in wanted]
        return events

    def clear_events(self, parent_run_id: str, run_ids: list[str] | None = None) -> list[dict[str, Any]]:
        parent = self.load(parent_run_id)
        if not parent:
            return []
        if run_ids:
            wanted = set(run_ids)
            kept = []
            taken = []
            for event in parent.queued_events:
                if event.get("run_id") in wanted:
                    taken.append(event)
                else:
                    kept.append(event)
            parent.queued_events = kept
        else:
            taken = list(parent.queued_events)
            parent.queued_events = []
        self._save(parent)
        return taken

    def set_wake_queued(self, parent_run_id: str, queued: bool) -> RunRecord | None:
        if not (run := self.load(parent_run_id)):
            return None
        run.wake_queued = queued
        return self._save(run)

    def record_wait_request(self, parent_run_id: str, request: WaitRequest) -> None:
        self._wait_requests[parent_run_id] = request

    def pop_wait_request(self, parent_run_id: str) -> WaitRequest | None:
        return self._wait_requests.pop(parent_run_id, None)
