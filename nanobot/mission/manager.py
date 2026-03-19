"""Global mission storage, planning, and execution helpers."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.config.paths import get_missions_dir
from nanobot.mission.prompts import MissionPromptBuilder
from nanobot.providers.base import LLMProvider
from nanobot.runs import ChildHandoff, RunManager
from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager

MISSION_STATUSES = {"planning", "approved", "running", "paused", "completed"}
FEATURE_ACTIVE_STATUSES = {"pending", "ready", "blocked", "partial", "failed"}
FEATURE_TERMINAL_STATUSES = {"completed"}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


@dataclass
class MissionMilestone:
    id: str
    title: str
    description: str = ""


@dataclass
class MissionValidationAssertion:
    id: str
    assertion: str
    rationale: str = ""


@dataclass
class ValidationAssertionState:
    id: str
    assertion: str
    rationale: str = ""
    status: str = "pending"
    claimed_by_feature_id: str = ""
    evidence: list[dict[str, str]] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now)


@dataclass
class MissionFeature:
    id: str
    description: str
    status: str = "pending"
    milestone: str = ""
    preconditions: list[str] = field(default_factory=list)
    expected_behavior: list[str] = field(default_factory=list)
    verification_steps: list[str] = field(default_factory=list)
    fulfills: list[str] = field(default_factory=list)
    skill_name: str = ""
    current_worker_run_id: str = ""
    completed_worker_run_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionFeature":
        return cls(
            id=str(data.get("id") or "").strip(),
            description=str(data.get("description") or "").strip(),
            status=str(data.get("status") or "pending"),
            milestone=str(data.get("milestone") or ""),
            preconditions=[str(item) for item in _as_list(data.get("preconditions")) if str(item).strip()],
            expected_behavior=[
                str(item) for item in _as_list(data.get("expected_behavior")) if str(item).strip()
            ],
            verification_steps=[
                str(item) for item in _as_list(data.get("verification_steps")) if str(item).strip()
            ],
            fulfills=[str(item) for item in _as_list(data.get("fulfills")) if str(item).strip()],
            skill_name=str(data.get("skill_name") or ""),
            current_worker_run_id=str(data.get("current_worker_run_id") or ""),
            completed_worker_run_id=str(data.get("completed_worker_run_id") or ""),
        )


@dataclass
class MissionInstruction:
    text: str
    sender_id: str
    timestamp: str


@dataclass
class MissionState:
    mission_id: str
    title: str
    status: str
    scope_key: str
    channel: str
    chat_id: str
    message_thread_id: int | None
    workspace: str
    created_at: str
    updated_at: str
    mission_summary: str
    latest_instruction: str
    mission_run_id: str = ""
    approval_required: bool = True
    approved_at: str = ""
    started_at: str = ""
    paused_at: str = ""
    completed_at: str = ""
    total_features: int = 0
    ready_feature_count: int = 0
    running_feature_count: int = 0
    completed_feature_count: int = 0
    failed_feature_count: int = 0
    active_worker_run_ids: list[str] = field(default_factory=list)
    orchestrator_return_pending: bool = False
    last_return_reason: str = ""
    progress_event_count: int = 0
    last_worker_batch_at: str = ""


@dataclass
class MissionHandoff:
    feature_id: str
    worker_run_id: str
    completion_status: str
    return_to_orchestrator: bool
    summary: str
    concrete_implemented_work: str = ""
    unfinished_work: str = ""
    verification_evidence: list[dict[str, str]] = field(default_factory=list)
    discovered_issues: list[dict[str, str]] = field(default_factory=list)
    next_recommended_action: str = ""
    timestamp: str = field(default_factory=_utc_now)

    @classmethod
    def from_child_event(
        cls,
        feature_id: str,
        worker_run_id: str,
        payload: Any,
    ) -> "MissionHandoff":
        handoff = ChildHandoff.normalize(payload)
        return cls(
            feature_id=feature_id,
            worker_run_id=worker_run_id,
            completion_status=handoff.status,
            return_to_orchestrator=handoff.return_to_orchestrator,
            summary=handoff.summary,
            concrete_implemented_work=handoff.what_was_done,
            unfinished_work=handoff.what_remains,
            verification_evidence=handoff.evidence,
            discovered_issues=handoff.discovered_issues,
            next_recommended_action=handoff.next_action,
        )


@dataclass
class Mission:
    mission_id: str
    title: str
    status: str
    scope_key: str
    channel: str
    chat_id: str
    message_thread_id: int | None
    workspace: str
    created_at: str
    updated_at: str
    mission_summary: str
    latest_instruction: str
    mission_run_id: str = ""
    approval_required: bool = True
    approved_at: str = ""
    started_at: str = ""
    paused_at: str = ""
    completed_at: str = ""
    active_worker_run_ids: list[str] = field(default_factory=list)
    orchestrator_return_pending: bool = False
    last_return_reason: str = ""
    progress_event_count: int = 0
    last_worker_batch_at: str = ""
    features: list[MissionFeature] = field(default_factory=list)
    milestones: list[MissionMilestone] = field(default_factory=list)
    validation_assertions: list[MissionValidationAssertion] = field(default_factory=list)
    validation_state: list[ValidationAssertionState] = field(default_factory=list)
    execution_constraints: list[str] = field(default_factory=list)
    instructions: list[MissionInstruction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Mission":
        mission = cls(
            mission_id=data["mission_id"],
            title=data["title"],
            status=data.get("status", "planning"),
            scope_key=data["scope_key"],
            channel=data["channel"],
            chat_id=data["chat_id"],
            message_thread_id=data.get("message_thread_id"),
            workspace=data["workspace"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            mission_summary=data.get("mission_summary", ""),
            latest_instruction=data.get("latest_instruction", ""),
            mission_run_id=str(data.get("mission_run_id") or ""),
            approval_required=bool(data.get("approval_required", True)),
            approved_at=str(data.get("approved_at") or ""),
            started_at=str(data.get("started_at") or ""),
            paused_at=str(data.get("paused_at") or ""),
            completed_at=str(data.get("completed_at") or ""),
            active_worker_run_ids=[str(item) for item in _as_list(data.get("active_worker_run_ids"))],
            orchestrator_return_pending=bool(data.get("orchestrator_return_pending", False)),
            last_return_reason=str(data.get("last_return_reason") or ""),
            progress_event_count=int(data.get("progress_event_count") or 0),
            last_worker_batch_at=str(data.get("last_worker_batch_at") or ""),
            features=[MissionFeature.from_dict(item) for item in data.get("features", [])],
            milestones=[MissionMilestone(**item) for item in data.get("milestones", [])],
            validation_assertions=[
                MissionValidationAssertion(**item) for item in data.get("validation_assertions", [])
            ],
            validation_state=[
                ValidationAssertionState(**item) for item in data.get("validation_state", [])
            ],
            execution_constraints=[
                str(item) for item in _as_list(data.get("execution_constraints")) if str(item).strip()
            ],
            instructions=[MissionInstruction(**item) for item in data.get("instructions", [])],
        )
        mission.ensure_defaults()
        return mission

    def ensure_defaults(self) -> None:
        if self.status not in MISSION_STATUSES:
            self.status = "planning"
        if not self.milestones:
            self.milestones = [
                MissionMilestone(id="ms-001", title="Primary milestone", description=self.latest_instruction)
            ]
        if not self.features:
            self.features = [
                MissionFeature(
                    id="feat-001",
                    description=self.latest_instruction or "Implement the requested mission goal.",
                    milestone=self.milestones[0].id,
                    expected_behavior=["The requested mission outcome is implemented."],
                    verification_steps=["Run targeted tests or checks for the requested behavior."],
                )
            ]
        if not self.validation_assertions:
            self.validation_assertions = [
                MissionValidationAssertion(
                    id="va-001",
                    assertion=self.latest_instruction or "The mission goal is implemented.",
                    rationale="Fallback assertion generated from the latest mission instruction.",
                )
            ]
        if not any(feature.fulfills for feature in self.features):
            self.features[0].fulfills = [self.validation_assertions[0].id]
        if not self.validation_state:
            self.validation_state = [
                ValidationAssertionState(
                    id=item.id,
                    assertion=item.assertion,
                    rationale=item.rationale,
                )
                for item in self.validation_assertions
            ]
        self._sync_validation_claims()

    def _sync_validation_claims(self) -> None:
        claims: dict[str, str] = {}
        for feature in self.features:
            for assertion_id in feature.fulfills:
                claims[assertion_id] = feature.id
        state_by_id = {item.id: item for item in self.validation_state}
        synced: list[ValidationAssertionState] = []
        for assertion in self.validation_assertions:
            state = state_by_id.get(assertion.id) or ValidationAssertionState(
                id=assertion.id,
                assertion=assertion.assertion,
                rationale=assertion.rationale,
            )
            state.assertion = assertion.assertion
            state.rationale = assertion.rationale
            state.claimed_by_feature_id = claims.get(assertion.id, "")
            synced.append(state)
        self.validation_state = synced

    def counts(self) -> dict[str, int]:
        ready = sum(1 for feature in self.features if feature.status in {"pending", "ready", "blocked", "partial", "failed"} and not feature.current_worker_run_id and self.is_feature_ready(feature))
        running = sum(1 for feature in self.features if feature.current_worker_run_id)
        completed = sum(1 for feature in self.features if feature.status == "completed")
        failed = sum(1 for feature in self.features if feature.status == "failed")
        return {
            "total": len(self.features),
            "ready": ready,
            "running": running,
            "completed": completed,
            "failed": failed,
        }

    def is_feature_ready(self, feature: MissionFeature) -> bool:
        if feature.current_worker_run_id:
            return False
        if feature.status == "completed":
            return False
        completed = {item.id for item in self.features if item.status == "completed"}
        return all(precondition in completed for precondition in feature.preconditions)

    def to_state(self) -> MissionState:
        counts = self.counts()
        return MissionState(
            mission_id=self.mission_id,
            title=self.title,
            status=self.status,
            scope_key=self.scope_key,
            channel=self.channel,
            chat_id=self.chat_id,
            message_thread_id=self.message_thread_id,
            workspace=self.workspace,
            created_at=self.created_at,
            updated_at=self.updated_at,
            mission_summary=self.mission_summary,
            latest_instruction=self.latest_instruction,
            mission_run_id=self.mission_run_id,
            approval_required=self.approval_required,
            approved_at=self.approved_at,
            started_at=self.started_at,
            paused_at=self.paused_at,
            completed_at=self.completed_at,
            total_features=counts["total"],
            ready_feature_count=counts["ready"],
            running_feature_count=counts["running"],
            completed_feature_count=counts["completed"],
            failed_feature_count=counts["failed"],
            active_worker_run_ids=list(self.active_worker_run_ids),
            orchestrator_return_pending=self.orchestrator_return_pending,
            last_return_reason=self.last_return_reason,
            progress_event_count=self.progress_event_count,
            last_worker_batch_at=self.last_worker_batch_at,
        )


class MissionManager:
    """Manage globally-scoped missions with chat/topic ownership."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        base_dir: Path | None = None,
        runs: RunManager | None = None,
        subagents: "SubagentManager | None" = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_dir = ensure_dir(base_dir or get_missions_dir())
        self._active_file = self.base_dir / "active.json"
        self.runs = runs
        self.subagents = subagents

    @staticmethod
    def _mission_session_key(mission_id: str) -> str:
        return f"mission:{mission_id}"

    def get_active_mission(self, scope_key: str) -> Mission | None:
        mission_id = self._load_active().get(scope_key)
        if not mission_id:
            return None
        return self._load_mission(mission_id)

    async def clear_active_mission(self, scope_key: str) -> Mission | None:
        active = self._load_active()
        mission_id = active.pop(scope_key, None)
        if mission_id is None:
            return None
        self._save_active(active)
        mission = self._load_mission(mission_id)
        if mission is None:
            return None
        if self.subagents:
            for run_id in list(mission.active_worker_run_ids):
                await self.subagents.cancel_run(run_id)
        mission.active_worker_run_ids = []
        for feature in mission.features:
            feature.current_worker_run_id = ""
        if mission.mission_run_id and self.runs:
            self.runs.cancel_run(mission.mission_run_id)
        if mission.status in {"planning", "approved", "running"}:
            mission.status = "paused"
            mission.paused_at = _utc_now()
        mission.updated_at = _utc_now()
        self._append_progress_event(mission, "mission_scope_cleared", {"scope_key": scope_key})
        self._save_mission(mission)
        return mission

    def get_mission_by_run_id(self, run_id: str) -> Mission | None:
        for path in self.base_dir.glob("*/mission.json"):
            try:
                mission = Mission.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if mission.mission_run_id == run_id:
                return mission
        return None

    def owns_run(self, run_id: str) -> bool:
        return self.get_mission_by_run_id(run_id) is not None

    async def start_or_update(
        self,
        *,
        scope_key: str,
        channel: str,
        chat_id: str,
        message_thread_id: int | None,
        workspace: Path,
        goal: str,
        sender_id: str,
    ) -> Mission:
        mission = self.get_active_mission(scope_key)
        if mission is None:
            mission = self._create_empty_mission(
                scope_key=scope_key,
                channel=channel,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                workspace=workspace,
                goal=goal,
                sender_id=sender_id,
            )
        else:
            mission.instructions.append(
                MissionInstruction(text=goal, sender_id=sender_id, timestamp=_utc_now())
            )
            mission.latest_instruction = goal
            mission.updated_at = _utc_now()
            mission.status = "planning"
            mission.approval_required = True
            mission.approved_at = ""
            mission.paused_at = ""
            mission.completed_at = ""
            mission.orchestrator_return_pending = False
            mission.last_return_reason = ""
            if mission.active_worker_run_ids and self.subagents:
                for run_id in list(mission.active_worker_run_ids):
                    await self.subagents.cancel_run(run_id)
            mission.active_worker_run_ids = []
            for feature in mission.features:
                feature.current_worker_run_id = ""
            if mission.mission_run_id and self.runs:
                self.runs.cancel_run(mission.mission_run_id)
            mission.mission_run_id = ""

        planned = await self._plan_mission(mission)
        self._save_mission(planned)
        active = self._load_active()
        active[scope_key] = planned.mission_id
        self._save_active(active)
        self._append_progress_event(planned, "mission_planned", {"goal": goal})
        self._save_mission(planned)
        return planned

    async def approve(self, scope_key: str) -> Mission | None:
        mission = self.get_active_mission(scope_key)
        if mission is None:
            return None
        mission.approval_required = False
        if mission.status == "planning":
            mission.status = "approved"
        mission.approved_at = mission.approved_at or _utc_now()
        mission.updated_at = _utc_now()
        self._append_progress_event(mission, "mission_approved", {})
        self._save_mission(mission)
        return mission

    async def pause(self, scope_key: str) -> Mission | None:
        mission = self.get_active_mission(scope_key)
        if mission is None:
            return None
        if self.subagents:
            for run_id in list(mission.active_worker_run_ids):
                await self.subagents.cancel_run(run_id)
        mission.active_worker_run_ids = []
        mission.status = "paused"
        mission.paused_at = _utc_now()
        mission.updated_at = _utc_now()
        if mission.mission_run_id and self.runs:
            self.runs.set_status(mission.mission_run_id, "waiting")
        self._append_progress_event(mission, "mission_paused", {})
        self._save_mission(mission)
        return mission

    async def start_or_resume_execution(self, scope_key: str) -> Mission | None:
        mission = self.get_active_mission(scope_key)
        if mission is None:
            return None
        if mission.approval_required or mission.status == "planning":
            raise ValueError("Mission approval is required before execution can start.")
        if not self.runs or not self.subagents:
            raise ValueError("Mission runtime is unavailable without runs and subagent managers.")
        parent = self.runs.load(mission.mission_run_id) if mission.mission_run_id else None
        if not parent or parent.status == "cancelled":
            parent = self.runs.create_parent_run(
                session_key=self._mission_session_key(mission.mission_id),
                goal=f"mission:{mission.mission_id}",
                messages=[{"role": "system", "content": f"mission:{mission.mission_id}"}],
            )
            mission.mission_run_id = parent.run_id
            mission.started_at = mission.started_at or _utc_now()
        await self._advance_execution(mission)
        self._save_mission(mission)
        return mission

    async def resume_parent_run(self, parent_run_id: str) -> Mission | None:
        if not self.runs:
            return None
        mission = self.get_mission_by_run_id(parent_run_id)
        if mission is None:
            return None
        await self._advance_execution(mission)
        self._save_mission(mission)
        return mission

    def mission_help_text(self) -> str:
        return (
            "Mission mode manages structured orchestrator/worker execution.\n"
            "Use `/mission <goal>` to plan or update a mission.\n"
            "Use `/mission approve`, `/mission start`, `/mission resume`, `/mission pause`, or `/mission inspect`."
        )

    def format_summary(self, mission: Mission) -> str:
        state = mission.to_state()
        approval = "awaiting approval" if state.approval_required else "approved"
        lines = [
            f"Mission: {mission.title}",
            f"State: {mission.status} ({approval})",
            f"Summary: {mission.mission_summary}",
            (
                "Progress: "
                f"{state.completed_feature_count}/{state.total_features} completed, "
                f"{state.running_feature_count} running, "
                f"{state.ready_feature_count} ready"
            ),
        ]
        if mission.mission_run_id:
            lines.append(f"Mission run: {mission.mission_run_id}")
        if mission.execution_constraints:
            lines.append("Constraints:")
            lines.extend(f"- {item}" for item in mission.execution_constraints)
        if mission.features:
            lines.append("Features:")
            for idx, feature in enumerate(mission.features, start=1):
                owner = f" worker={feature.current_worker_run_id}" if feature.current_worker_run_id else ""
                milestone = f" milestone={feature.milestone}" if feature.milestone else ""
                lines.append(
                    f"{idx}. [{feature.status}] {feature.id}:{milestone}{owner} {feature.description}".rstrip()
                )
        return "\n".join(lines)

    def record_handoff(self, mission_id: str, handoff: MissionHandoff) -> None:
        mission_dir = self._mission_dir(mission_id)
        with open(mission_dir / "handoffs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(handoff), ensure_ascii=False) + "\n")
        handoff_name = f"{handoff.timestamp.replace(':', '-')}-{handoff.worker_run_id}-{handoff.feature_id}.json"
        (mission_dir / "handoffs" / handoff_name).write_text(
            json.dumps(asdict(handoff), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _create_empty_mission(
        self,
        *,
        scope_key: str,
        channel: str,
        chat_id: str,
        message_thread_id: int | None,
        workspace: Path,
        goal: str,
        sender_id: str,
    ) -> Mission:
        mission_id = str(uuid.uuid4())
        now = _utc_now()
        mission = Mission(
            mission_id=mission_id,
            title="New Mission",
            status="planning",
            scope_key=scope_key,
            channel=channel,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            workspace=str(workspace),
            created_at=now,
            updated_at=now,
            mission_summary="",
            latest_instruction=goal,
            features=[],
            instructions=[MissionInstruction(text=goal, sender_id=sender_id, timestamp=now)],
        )
        mission.ensure_defaults()
        mission_dir = self._mission_dir(mission_id)
        ensure_dir(mission_dir / "artifacts")
        ensure_dir(mission_dir / "handoffs")
        (mission_dir / "handoffs.jsonl").touch()
        (mission_dir / "progress_log.jsonl").touch()
        self._save_mission(mission)
        return mission

    async def _plan_mission(self, mission: Mission) -> Mission:
        prompt = MissionPromptBuilder.build_orchestrator_prompt(Path(mission.workspace))
        request = self._planning_request(mission)
        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": request},
            ],
            model=self.model,
        )
        if response.finish_reason == "error":
            logger.warning("Mission planning failed for {}: {}", mission.mission_id, response.content)
            return self._fallback_plan(mission)

        parsed = self._parse_plan_response(response.content or "")
        if not parsed:
            logger.warning("Mission planning returned invalid JSON for {}", mission.mission_id)
            return self._fallback_plan(mission)

        mission.title = parsed["mission_title"]
        mission.mission_summary = parsed["mission_summary"]
        mission.status = "planning"
        mission.approval_required = True
        mission.approved_at = ""
        mission.completed_at = ""
        mission.paused_at = ""
        mission.milestones = [MissionMilestone(**item) for item in parsed["milestones"]]
        mission.validation_assertions = [
            MissionValidationAssertion(**item) for item in parsed["validation_assertions"]
        ]
        mission.execution_constraints = [
            str(item) for item in parsed.get("execution_constraints", []) if str(item).strip()
        ]
        mission.features = [MissionFeature.from_dict(item) for item in parsed["features"]]
        mission.validation_state = [
            ValidationAssertionState(
                id=item.id,
                assertion=item.assertion,
                rationale=item.rationale,
            )
            for item in mission.validation_assertions
        ]
        mission.ensure_defaults()
        self._validate_assertion_coverage(mission)
        mission.updated_at = _utc_now()
        self._write_mission_files(mission)
        return mission

    def _fallback_plan(self, mission: Mission) -> Mission:
        title = mission.latest_instruction[:60].strip() or "Mission"
        mission.title = title
        mission.mission_summary = mission.latest_instruction
        mission.status = "planning"
        mission.approval_required = True
        mission.milestones = [
            MissionMilestone(id="ms-001", title="Primary milestone", description=mission.latest_instruction)
        ]
        mission.validation_assertions = [
            MissionValidationAssertion(
                id="va-001",
                assertion=mission.latest_instruction or "The requested mission outcome is implemented.",
                rationale="Fallback validation coverage for the mission goal.",
            )
        ]
        mission.execution_constraints = ["Require explicit approval before execution starts."]
        mission.features = [
            MissionFeature(
                id="feat-001",
                description=mission.latest_instruction,
                status="pending",
                milestone="ms-001",
                expected_behavior=["The requested mission outcome is implemented."],
                verification_steps=["Run the most relevant automated tests or focused checks."],
                fulfills=["va-001"],
            )
        ]
        mission.validation_state = [
            ValidationAssertionState(
                id="va-001",
                assertion=mission.validation_assertions[0].assertion,
                rationale=mission.validation_assertions[0].rationale,
                claimed_by_feature_id="feat-001",
            )
        ]
        mission.updated_at = _utc_now()
        self._write_mission_files(mission)
        return mission

    def _planning_request(self, mission: Mission) -> str:
        transcript = "\n".join(
            f"- [{item.timestamp}] {item.sender_id}: {item.text}" for item in mission.instructions
        )
        return (
            "Plan this mission for nanobot.\n\n"
            f"Mission scope key: {mission.scope_key}\n"
            f"Workspace: {mission.workspace}\n"
            "Current mission requirements:\n"
            f"{transcript}\n\n"
            "Return planning JSON only."
        )

    def _parse_plan_response(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not all(k in data for k in ("mission_title", "mission_summary", "features")):
            return None
        milestones = data.get("milestones") or [
            {"id": "ms-001", "title": "Primary milestone", "description": data["mission_summary"]}
        ]
        features = []
        raw_features = _as_list(data.get("features"))
        for idx, item in enumerate(raw_features, start=1):
            if not isinstance(item, dict):
                continue
            feature_id = str(item.get("id") or f"feat-{idx:03d}")
            fulfills = [str(val) for val in _as_list(item.get("fulfills")) if str(val).strip()]
            features.append(
                {
                    "id": feature_id,
                    "description": str(item.get("description") or "").strip(),
                    "status": str(item.get("status") or "pending"),
                    "milestone": str(item.get("milestone") or milestones[0]["id"]),
                    "preconditions": [
                        str(val) for val in _as_list(item.get("preconditions")) if str(val).strip()
                    ],
                    "expected_behavior": [
                        str(val) for val in _as_list(item.get("expected_behavior")) if str(val).strip()
                    ],
                    "verification_steps": [
                        str(val) for val in _as_list(item.get("verification_steps")) if str(val).strip()
                    ],
                    "fulfills": fulfills,
                    "skill_name": str(item.get("skill_name") or ""),
                    "current_worker_run_id": "",
                    "completed_worker_run_id": "",
                }
            )
        assertions = _as_list(data.get("validation_assertions"))
        if not assertions:
            assertions = []
            for idx, feature in enumerate(features, start=1):
                assertion_id = f"va-{idx:03d}"
                feature["fulfills"] = feature["fulfills"] or [assertion_id]
                assertions.append(
                    {
                        "id": assertion_id,
                        "assertion": feature["description"] or f"Feature {feature['id']} is implemented.",
                        "rationale": "Fallback assertion derived from the feature plan.",
                    }
                )
        parsed = {
            "mission_title": str(data["mission_title"]).strip(),
            "mission_summary": str(data["mission_summary"]).strip(),
            "milestones": [
                {
                    "id": str(item.get("id") or f"ms-{idx:03d}"),
                    "title": str(item.get("title") or f"Milestone {idx}"),
                    "description": str(item.get("description") or ""),
                }
                for idx, item in enumerate(milestones, start=1)
                if isinstance(item, dict)
            ],
            "validation_assertions": [
                {
                    "id": str(item.get("id") or f"va-{idx:03d}"),
                    "assertion": str(item.get("assertion") or ""),
                    "rationale": str(item.get("rationale") or ""),
                }
                for idx, item in enumerate(assertions, start=1)
                if isinstance(item, dict)
            ],
            "execution_constraints": [
                str(item) for item in _as_list(data.get("execution_constraints")) if str(item).strip()
            ],
            "features": features,
        }
        return parsed if parsed["features"] else None

    def _validate_assertion_coverage(self, mission: Mission) -> None:
        claim_counts = {item.id: 0 for item in mission.validation_assertions}
        for feature in mission.features:
            for assertion_id in feature.fulfills:
                if assertion_id not in claim_counts:
                    raise ValueError(f"Feature {feature.id} references unknown validation assertion {assertion_id}.")
                claim_counts[assertion_id] += 1
        missing = [assertion_id for assertion_id, count in claim_counts.items() if count == 0]
        duplicated = [assertion_id for assertion_id, count in claim_counts.items() if count > 1]
        if missing or duplicated:
            problems: list[str] = []
            if missing:
                problems.append(f"unclaimed assertions: {', '.join(missing)}")
            if duplicated:
                problems.append(f"multiply claimed assertions: {', '.join(duplicated)}")
            raise ValueError("Validation contract coverage failed: " + "; ".join(problems))
        mission.ensure_defaults()

    def _write_mission_files(self, mission: Mission) -> None:
        mission.ensure_defaults()
        mission._sync_validation_claims()
        mission_dir = self._mission_dir(mission.mission_id)
        ensure_dir(mission_dir / "artifacts")
        ensure_dir(mission_dir / "handoffs")
        (mission_dir / "handoffs.jsonl").touch()
        (mission_dir / "progress_log.jsonl").touch()
        (mission_dir / "AGENTS.md").write_text(
            self._mission_agents_content(mission),
            encoding="utf-8",
        )
        (mission_dir / "mission.md").write_text(
            self._mission_markdown(mission),
            encoding="utf-8",
        )
        (mission_dir / "state.json").write_text(
            json.dumps(asdict(mission.to_state()), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (mission_dir / "features.json").write_text(
            json.dumps({"features": [asdict(feature) for feature in mission.features]}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        (mission_dir / "validation-contract.md").write_text(
            self._validation_contract_markdown(mission),
            encoding="utf-8",
        )
        (mission_dir / "validation-state.json").write_text(
            json.dumps(
                {"assertions": [asdict(item) for item in mission.validation_state]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def _mission_markdown(self, mission: Mission) -> str:
        lines = [f"# {mission.title}", "", mission.mission_summary, ""]
        if mission.milestones:
            lines.extend(["## Milestones", ""])
            for item in mission.milestones:
                lines.append(f"- {item.id}: {item.title} — {item.description}".rstrip(" —"))
            lines.append("")
        if mission.execution_constraints:
            lines.extend(["## Execution Constraints", ""])
            lines.extend(f"- {item}" for item in mission.execution_constraints)
            lines.append("")
        if mission.features:
            lines.extend(["## Features", ""])
            for item in mission.features:
                lines.append(f"- {item.id}: {item.description}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _validation_contract_markdown(self, mission: Mission) -> str:
        lines = ["# Validation Contract", ""]
        for assertion in mission.validation_assertions:
            claimed_by = next(
                (feature.id for feature in mission.features if assertion.id in feature.fulfills),
                "UNCLAIMED",
            )
            lines.append(f"## {assertion.id}")
            lines.append(assertion.assertion)
            lines.append("")
            if assertion.rationale:
                lines.append(f"Rationale: {assertion.rationale}")
            lines.append(f"Claimed by feature: {claimed_by}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _mission_agents_content(self, mission: Mission) -> str:
        mission_dir = self._mission_dir(mission.mission_id)
        return (
            "# Mission Runtime Rules\n\n"
            "Canonical mission truth is orchestrator-owned.\n\n"
            f"- Workers must not mutate `{mission_dir / 'features.json'}`.\n"
            f"- Workers must not mutate `{mission_dir / 'state.json'}`.\n"
            f"- Workers must not mutate `{mission_dir / 'validation-contract.md'}`.\n"
            f"- Workers must not mutate `{mission_dir / 'validation-state.json'}`.\n"
            f"- Workers must not mutate `{mission_dir / 'AGENTS.md'}`.\n"
        )

    def _save_mission(self, mission: Mission) -> None:
        mission.ensure_defaults()
        mission_dir = self._mission_dir(mission.mission_id)
        ensure_dir(mission_dir)
        payload = {
            **asdict(mission.to_state()),
            "features": [asdict(feature) for feature in mission.features],
            "milestones": [asdict(item) for item in mission.milestones],
            "validation_assertions": [asdict(item) for item in mission.validation_assertions],
            "validation_state": [asdict(item) for item in mission.validation_state],
            "execution_constraints": list(mission.execution_constraints),
            "instructions": [asdict(item) for item in mission.instructions],
        }
        (mission_dir / "mission.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._write_mission_files(mission)

    def _load_mission(self, mission_id: str) -> Mission | None:
        path = self._mission_dir(mission_id) / "mission.json"
        if not path.exists():
            return None
        try:
            return Mission.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Failed to load mission {}: {}", mission_id, exc)
            return None

    def _load_active(self) -> dict[str, str]:
        if not self._active_file.exists():
            return {}
        try:
            return json.loads(self._active_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_active(self, data: dict[str, str]) -> None:
        self._active_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _mission_dir(self, mission_id: str) -> Path:
        return self.base_dir / mission_id

    def _append_progress_event(self, mission: Mission, event_type: str, payload: dict[str, Any]) -> None:
        mission.progress_event_count += 1
        mission.updated_at = _utc_now()
        mission_dir = self._mission_dir(mission.mission_id)
        ensure_dir(mission_dir)
        with open(mission_dir / "progress_log.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": _utc_now(),
                        "event_type": event_type,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    async def _advance_execution(self, mission: Mission) -> None:
        if not self.runs or not self.subagents:
            raise ValueError("Mission runtime is unavailable without runs and subagent managers.")
        if not mission.mission_run_id:
            raise ValueError("Mission has no orchestrator run id.")

        events = self.runs.clear_events(mission.mission_run_id)
        if events:
            self.runs.set_wake_queued(mission.mission_run_id, False)
            await self._apply_worker_events(mission, events)
        self._reconcile_worker_ownership(mission)

        if all(feature.status == "completed" for feature in mission.features):
            mission.status = "completed"
            mission.completed_at = mission.completed_at or _utc_now()
            mission.active_worker_run_ids = []
            self.runs.set_status(mission.mission_run_id, "completed")
            self._append_progress_event(mission, "mission_completed", {})
            return

        ready_features = [feature for feature in mission.features if mission.is_feature_ready(feature)]
        ready_ids = [feature.id for feature in ready_features]
        running_ids = list(mission.active_worker_run_ids)
        decision = await self._choose_dispatch(mission, ready_ids, running_ids, events)

        if decision["mission_status"] == "completed" and not running_ids:
            mission.status = "completed"
            mission.completed_at = mission.completed_at or _utc_now()
            self.runs.set_status(mission.mission_run_id, "completed")
            self._append_progress_event(mission, "mission_completed", {"summary": decision["summary"]})
            return

        if decision["mission_status"] == "paused":
            mission.status = "paused"
            mission.paused_at = _utc_now()
            self.runs.set_status(mission.mission_run_id, "waiting")
            self._append_progress_event(
                mission,
                "mission_paused_by_orchestrator",
                {"reason": decision["pause_reason"], "summary": decision["summary"]},
            )
            return

        dispatched_run_ids = await self._dispatch_features(
            mission,
            [feature for feature in ready_features if feature.id in set(decision["dispatch_feature_ids"])],
        )
        mission.status = "running"
        mission.paused_at = ""
        mission.started_at = mission.started_at or _utc_now()
        mission.updated_at = _utc_now()
        mission.active_worker_run_ids = [
            feature.current_worker_run_id for feature in mission.features if feature.current_worker_run_id
        ]
        if dispatched_run_ids or mission.active_worker_run_ids:
            self.runs.mark_waiting(
                mission.mission_run_id,
                [{"role": "system", "content": f"mission:{mission.mission_id}"}],
                mission.active_worker_run_ids,
            )
        else:
            self.runs.set_status(mission.mission_run_id, "running")
        if dispatched_run_ids:
            self._append_progress_event(
                mission,
                "workers_dispatched",
                {"feature_ids": [feature.id for feature in mission.features if feature.current_worker_run_id in dispatched_run_ids]},
            )

    async def _apply_worker_events(self, mission: Mission, events: list[dict[str, Any]]) -> None:
        by_run_id = {feature.current_worker_run_id: feature for feature in mission.features if feature.current_worker_run_id}
        mission.last_worker_batch_at = _utc_now()
        mission.orchestrator_return_pending = False
        mission.last_return_reason = ""
        for event in events:
            run_id = str(event.get("run_id") or "")
            feature = by_run_id.get(run_id)
            if feature is None:
                continue
            handoff = MissionHandoff.from_child_event(
                feature.id,
                run_id,
                event.get("handoff"),
            )
            self.record_handoff(mission.mission_id, handoff)
            self._append_progress_event(
                mission,
                "worker_handoff",
                {
                    "feature_id": feature.id,
                    "worker_run_id": run_id,
                    "status": handoff.completion_status,
                    "return_to_orchestrator": handoff.return_to_orchestrator,
                },
            )
            feature.current_worker_run_id = ""
            feature.completed_worker_run_id = run_id
            if handoff.completion_status == "success" and not handoff.unfinished_work and not handoff.discovered_issues:
                feature.status = "completed"
            elif handoff.completion_status == "success":
                feature.status = "partial"
            elif handoff.completion_status == "partial":
                feature.status = "partial"
            else:
                feature.status = "failed"
            if handoff.return_to_orchestrator or handoff.completion_status != "success":
                mission.orchestrator_return_pending = True
                mission.last_return_reason = handoff.summary
            self._update_validation_state_for_feature(mission, feature, handoff)
        mission.active_worker_run_ids = [
            feature.current_worker_run_id for feature in mission.features if feature.current_worker_run_id
        ]

    def _update_validation_state_for_feature(
        self,
        mission: Mission,
        feature: MissionFeature,
        handoff: MissionHandoff,
    ) -> None:
        state_by_id = {item.id: item for item in mission.validation_state}
        new_status = "implemented" if feature.status == "completed" else "pending"
        for assertion_id in feature.fulfills:
            if assertion_id not in state_by_id:
                continue
            item = state_by_id[assertion_id]
            item.claimed_by_feature_id = feature.id
            item.status = new_status
            item.updated_at = _utc_now()
            if handoff.verification_evidence:
                item.evidence = handoff.verification_evidence

    async def _choose_dispatch(
        self,
        mission: Mission,
        ready_ids: list[str],
        running_ids: list[str],
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        default = {
            "mission_status": (
                "completed"
                if not ready_ids and not running_ids
                else "paused"
                if ready_ids and not running_ids
                else "running"
            ),
            "dispatch_feature_ids": [],
            "summary": (
                "No ready features remain."
                if not ready_ids and not running_ids
                else "Mission is paused until the orchestrator can make an explicit dispatch decision."
                if ready_ids and not running_ids
                else "Continue waiting for active workers."
            ),
            "pause_reason": (
                (mission.last_return_reason or "Orchestrator decision unavailable.")
                if ready_ids or mission.orchestrator_return_pending
                else ""
            ),
            "return_to_user": False,
            "user_message": "",
        }
        prompt = MissionPromptBuilder.build_runner_prompt(Path(mission.workspace))
        request = json.dumps(
            {
                "mission_id": mission.mission_id,
                "mission_title": mission.title,
                "mission_summary": mission.mission_summary,
                "status": mission.status,
                "execution_constraints": mission.execution_constraints,
                "ready_feature_ids": ready_ids,
                "running_worker_run_ids": running_ids,
                "orchestrator_return_pending": mission.orchestrator_return_pending,
                "latest_return_reason": mission.last_return_reason,
                "features": [asdict(feature) for feature in mission.features],
                "latest_worker_events": events,
            },
            ensure_ascii=False,
        )
        try:
            response = await self.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request},
                ],
                model=self.model,
            )
        except Exception:
            return default
        if response.finish_reason == "error":
            return default
        parsed = self._parse_runner_response(response.content or "")
        if not parsed:
            return default
        parsed["dispatch_feature_ids"] = [
            feature_id for feature_id in parsed["dispatch_feature_ids"] if feature_id in ready_ids
        ]
        if parsed["mission_status"] == "running" and mission.orchestrator_return_pending and not parsed["dispatch_feature_ids"]:
            parsed["mission_status"] = "paused"
            parsed["pause_reason"] = parsed["pause_reason"] or mission.last_return_reason or "Worker requested orchestrator review."
        if not ready_ids and not running_ids and parsed["mission_status"] == "running":
            parsed["mission_status"] = "completed"
        return parsed

    def _parse_runner_response(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        status = str(data.get("mission_status") or "running")
        if status not in {"running", "paused", "completed"}:
            return None
        return {
            "mission_status": status,
            "dispatch_feature_ids": [
                str(item) for item in _as_list(data.get("dispatch_feature_ids")) if str(item).strip()
            ],
            "summary": str(data.get("summary") or ""),
            "pause_reason": str(data.get("pause_reason") or ""),
            "return_to_user": bool(data.get("return_to_user", False)),
            "user_message": str(data.get("user_message") or ""),
        }

    async def _dispatch_features(self, mission: Mission, features: list[MissionFeature]) -> list[str]:
        if not features:
            return []
        if not self.subagents:
            raise ValueError("Mission runtime cannot dispatch workers without a subagent manager.")
        mission_dir = self._mission_dir(mission.mission_id)
        dispatched: list[str] = []
        for feature in features:
            if feature.current_worker_run_id or not mission.is_feature_ready(feature):
                continue
            task = self._worker_task_payload(mission, feature)
            system_prompt = MissionPromptBuilder.build_worker_prompt(
                Path(mission.workspace),
                mission_dir,
                feature.id,
                feature.description,
            )
            result = await self.subagents.spawn(
                task=task,
                label=feature.id,
                origin_channel=mission.channel,
                origin_chat_id=mission.chat_id,
                session_key=self._mission_session_key(mission.mission_id),
                parent_run_id=mission.mission_run_id,
                system_prompt=system_prompt,
                protected_paths=[
                    str(mission_dir / "features.json"),
                    str(mission_dir / "state.json"),
                    str(mission_dir / "validation-contract.md"),
                    str(mission_dir / "validation-state.json"),
                    str(mission_dir / "AGENTS.md"),
                ],
            )
            feature.status = "running"
            feature.current_worker_run_id = result["run_id"]
            dispatched.append(result["run_id"])
        return dispatched

    def _worker_task_payload(self, mission: Mission, feature: MissionFeature) -> str:
        payload = {
            "mission_id": mission.mission_id,
            "mission_title": mission.title,
            "mission_summary": mission.mission_summary,
            "feature": asdict(feature),
            "milestones": [asdict(item) for item in mission.milestones],
            "execution_constraints": mission.execution_constraints,
            "mission_dir": str(self._mission_dir(mission.mission_id)),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _reconcile_worker_ownership(self, mission: Mission) -> None:
        if not self.runs:
            return
        active: list[str] = []
        for feature in mission.features:
            run_id = feature.current_worker_run_id
            if not run_id:
                continue
            record = self.runs.load(run_id)
            if record is None:
                feature.current_worker_run_id = ""
                if feature.status == "running":
                    feature.status = "pending"
                continue
            if record.status in {"cancelled", "completed", "failed"}:
                feature.current_worker_run_id = ""
                if feature.status == "running":
                    feature.status = "pending"
                continue
            active.append(run_id)
        mission.active_worker_run_ids = active
