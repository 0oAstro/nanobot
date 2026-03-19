from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.mission.manager import MissionManager
from nanobot.mission.prompts import MissionPromptBuilder
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.runs import ChildHandoff, RunManager
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus


class _QueuedProvider:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if self.contents else ""
        return LLMResponse(content=content)


class _FakeSubagents:
    def __init__(self, runs: RunManager | None = None) -> None:
        self.spawn_calls: list[dict] = []
        self.cancelled: list[str] = []
        self._next_id = 0
        self.runs = runs

    async def spawn(self, **kwargs):
        self.spawn_calls.append(kwargs)
        self._next_id += 1
        run_id = f"worker-{self._next_id:03d}"
        if self.runs and kwargs.get("parent_run_id"):
            child = self.runs.create_subagent_run(
                parent_run_id=kwargs["parent_run_id"],
                session_key=kwargs.get("session_key", "cli:direct"),
                goal=kwargs.get("task", ""),
                label=kwargs.get("label", ""),
            )
            run_id = child.run_id
        return {"run_id": run_id, "label": kwargs.get("label", "")}

    async def cancel_run(self, run_id: str) -> bool:
        self.cancelled.append(run_id)
        return True


class _SubagentProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    def get_default_model(self) -> str:
        return "test-model"

    async def chat_with_retry(self, **kwargs):
        if not self.responses:
            raise AssertionError("No subagent response queued")
        return self.responses.pop(0)


def _planning_payload(*, preconditions: list[str] | None = None) -> str:
    return json.dumps(
        {
            "mission_title": "Mission Title",
            "mission_summary": "Mission summary",
            "state": "planning",
            "milestones": [
                {"id": "ms-001", "title": "Plan", "description": "Initial milestone"},
                {"id": "ms-002", "title": "Finish", "description": "Follow-up milestone"},
            ],
            "validation_assertions": [
                {"id": "va-001", "assertion": "Feature one works", "rationale": "required"},
                {"id": "va-002", "assertion": "Feature two works", "rationale": "required"},
            ],
            "execution_constraints": ["Keep approval explicit."],
            "features": [
                {
                    "id": "feat-001",
                    "description": "Implement feature one",
                    "status": "pending",
                    "milestone": "ms-001",
                    "preconditions": [],
                    "expected_behavior": ["It works"],
                    "verification_steps": ["Run test one"],
                    "fulfills": ["va-001"],
                    "skill_name": "",
                },
                {
                    "id": "feat-002",
                    "description": "Implement feature two",
                    "status": "pending",
                    "milestone": "ms-002",
                    "preconditions": preconditions or [],
                    "expected_behavior": ["It also works"],
                    "verification_steps": ["Run test two"],
                    "fulfills": ["va-002"],
                    "skill_name": "",
                },
            ],
        }
    )


@pytest.mark.asyncio
async def test_start_or_update_creates_richer_mission_artifacts(tmp_path: Path) -> None:
    provider = _QueuedProvider(_planning_payload())
    manager = MissionManager(provider=provider, model="test-model", base_dir=tmp_path)

    mission = await manager.start_or_update(
        scope_key="telegram:123",
        channel="telegram",
        chat_id="123",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )

    mission_dir = tmp_path / mission.mission_id
    assert mission_dir.exists()
    assert (mission_dir / "mission.json").exists()
    assert (mission_dir / "mission.md").exists()
    assert (mission_dir / "state.json").exists()
    assert (mission_dir / "features.json").exists()
    assert (mission_dir / "validation-contract.md").exists()
    assert (mission_dir / "validation-state.json").exists()
    assert (mission_dir / "handoffs.jsonl").exists()
    assert (mission_dir / "handoffs").is_dir()
    assert (mission_dir / "progress_log.jsonl").exists()
    assert (mission_dir / "AGENTS.md").exists()

    state = json.loads((mission_dir / "state.json").read_text(encoding="utf-8"))
    validation_state = json.loads((mission_dir / "validation-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "planning"
    assert state["approval_required"] is True
    assert len(validation_state["assertions"]) == 2
    assert all(item["status"] == "pending" for item in validation_state["assertions"])
    assert manager.get_active_mission("telegram:123") is not None


@pytest.mark.asyncio
async def test_start_or_update_appends_instruction_for_active_scope(tmp_path: Path) -> None:
    provider = _QueuedProvider(_planning_payload(), _planning_payload())
    manager = MissionManager(provider=provider, model="test-model", base_dir=tmp_path)

    first = await manager.start_or_update(
        scope_key="telegram:123",
        channel="telegram",
        chat_id="123",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    second = await manager.start_or_update(
        scope_key="telegram:123",
        channel="telegram",
        chat_id="123",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Also cover Telegram",
        sender_id="u2",
    )

    assert first.mission_id == second.mission_id
    assert second.instructions[-1].text == "Also cover Telegram"
    assert second.status == "planning"
    assert second.approval_required is True


@pytest.mark.asyncio
async def test_start_requires_explicit_approval(tmp_path: Path) -> None:
    provider = _QueuedProvider(_planning_payload())
    runs = RunManager(tmp_path / "workspace")
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=_FakeSubagents(runs),
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )

    with pytest.raises(ValueError, match="approval"):
        await manager.start_or_resume_execution("cli:direct")


@pytest.mark.asyncio
async def test_clear_active_mission_removes_scope_and_pauses_stale_plan(tmp_path: Path) -> None:
    provider = _QueuedProvider(_planning_payload())
    manager = MissionManager(provider=provider, model="test-model", base_dir=tmp_path / "missions")

    mission = await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )

    cleared = await manager.clear_active_mission("cli:direct")

    assert cleared is not None
    assert cleared.mission_id == mission.mission_id
    assert manager.get_active_mission("cli:direct") is None

    reloaded = MissionManager(provider=provider, model="test-model", base_dir=tmp_path / "missions")
    stored = reloaded._load_mission(mission.mission_id)
    assert stored is not None
    assert stored.status == "paused"

    progress = (tmp_path / "missions" / mission.mission_id / "progress_log.jsonl").read_text(encoding="utf-8")
    assert '"event_type": "mission_scope_cleared"' in progress


@pytest.mark.asyncio
async def test_start_dispatches_ready_features_and_tracks_parent_run(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        _planning_payload(),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001"],
                "summary": "Start the first feature.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
    )
    runs = RunManager(tmp_path / "workspace")
    subagents = _FakeSubagents(runs)
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=subagents,
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")

    assert mission is not None
    assert mission.mission_run_id
    assert mission.status == "running"
    assert mission.features[0].current_worker_run_id
    assert mission.features[1].current_worker_run_id == ""
    assert subagents.spawn_calls[0]["parent_run_id"] == mission.mission_run_id
    assert subagents.spawn_calls[0]["session_key"] == f"mission:{mission.mission_id}"
    parent = runs.load(mission.mission_run_id)
    assert parent is not None
    assert parent.session_key == f"mission:{mission.mission_id}"
    assert parent.status == "waiting"
    assert parent.waiting_on == [mission.features[0].current_worker_run_id]


@pytest.mark.asyncio
async def test_parallel_dispatch_uses_feature_ownership_to_prevent_duplicates(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        _planning_payload(),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001", "feat-002"],
                "summary": "Run both ready features.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001", "feat-002"],
                "summary": "Still running.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
    )
    runs = RunManager(tmp_path / "workspace")
    subagents = _FakeSubagents(runs)
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=subagents,
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None
    first_run_ids = {feature.current_worker_run_id for feature in mission.features}
    assert len(first_run_ids) == 2
    assert all(first_run_ids)

    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None
    assert len(subagents.spawn_calls) == 2
    assert {feature.current_worker_run_id for feature in mission.features} == first_run_ids


@pytest.mark.asyncio
async def test_worker_handoffs_update_files_and_feature_state(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        _planning_payload(preconditions=["feat-001"]),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001"],
                "summary": "Start feature one.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-002"],
                "summary": "Continue with feature two.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
    )
    runs = RunManager(tmp_path / "workspace")
    subagents = _FakeSubagents(runs)
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=subagents,
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None
    child = runs.create_subagent_run(
        parent_run_id=mission.mission_run_id,
        session_key="cli:direct",
        goal="child task",
        label="feat-001",
    )
    mission.features[0].current_worker_run_id = child.run_id
    mission.active_worker_run_ids = [child.run_id]
    manager._save_mission(mission)

    runs.complete_subagent(
        child.run_id,
        ChildHandoff.normalize(
            {
                "status": "success",
                "return_to_orchestrator": False,
                "summary": "feature one finished",
                "what_was_done": "implemented feature one",
                "what_remains": "",
                "evidence": [{"type": "test", "value": "uv run pytest tests/test_one.py -q", "note": "passes"}],
                "discovered_issues": [],
                "next_action": "start feature two",
            }
        ),
    )

    mission = await manager.resume_parent_run(mission.mission_run_id)
    assert mission is not None
    assert mission.features[0].status == "completed"
    assert mission.features[0].completed_worker_run_id == child.run_id
    assert mission.features[1].current_worker_run_id

    mission_dir = tmp_path / "missions" / mission.mission_id
    handoffs = (mission_dir / "handoffs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    progress = (mission_dir / "progress_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    handoff_files = list((mission_dir / "handoffs").glob("*.json"))
    validation_state = json.loads((mission_dir / "validation-state.json").read_text(encoding="utf-8"))
    assert len(handoffs) == 1
    assert len(handoff_files) == 1
    assert any('"event_type": "worker_handoff"' in line for line in progress)
    assert validation_state["assertions"][0]["status"] == "implemented"


@pytest.mark.asyncio
async def test_partial_or_return_to_orchestrator_pauses_for_review(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        _planning_payload(),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001"],
                "summary": "Start feature one.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
        json.dumps(
            {
                "mission_status": "paused",
                "dispatch_feature_ids": [],
                "summary": "Review the worker findings.",
                "pause_reason": "Worker reported a blocker.",
                "return_to_user": False,
                "user_message": "",
            }
        ),
    )
    runs = RunManager(tmp_path / "workspace")
    subagents = _FakeSubagents(runs)
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=subagents,
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None
    child = runs.create_subagent_run(
        parent_run_id=mission.mission_run_id,
        session_key="cli:direct",
        goal="child task",
        label="feat-001",
    )
    mission.features[0].current_worker_run_id = child.run_id
    mission.active_worker_run_ids = [child.run_id]
    manager._save_mission(mission)

    runs.complete_subagent(
        child.run_id,
        ChildHandoff.normalize(
            {
                "status": "partial",
                "return_to_orchestrator": True,
                "summary": "blocked on missing dependency",
                "what_was_done": "set up part of feature one",
                "what_remains": "finish integration",
                "evidence": [],
                "discovered_issues": [{"severity": "high", "description": "dependency missing"}],
                "next_action": "replan",
            }
        ),
    )

    mission = await manager.resume_parent_run(mission.mission_run_id)
    assert mission is not None
    assert mission.status == "paused"
    assert mission.orchestrator_return_pending is True
    assert mission.features[0].status == "partial"


@pytest.mark.asyncio
async def test_approve_does_not_downgrade_running_mission(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        _planning_payload(),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001"],
                "summary": "Start feature one.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
    )
    runs = RunManager(tmp_path / "workspace")
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=_FakeSubagents(runs),
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None
    assert mission.status == "running"

    mission = await manager.approve("cli:direct")
    assert mission is not None
    assert mission.status == "running"
    assert mission.approval_required is False


@pytest.mark.asyncio
async def test_pause_preserves_late_worker_handoff_until_resume(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        _planning_payload(),
        json.dumps(
            {
                "mission_status": "running",
                "dispatch_feature_ids": ["feat-001"],
                "summary": "Start feature one.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
        json.dumps(
            {
                "mission_status": "completed",
                "dispatch_feature_ids": [],
                "summary": "Mission complete.",
                "pause_reason": "",
                "return_to_user": False,
                "user_message": "",
            }
        ),
    )
    runs = RunManager(tmp_path / "workspace")
    subagents = _FakeSubagents(runs)
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=subagents,
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None

    child = runs.create_subagent_run(
        parent_run_id=mission.mission_run_id,
        session_key="cli:direct",
        goal="child task",
        label="feat-001",
    )
    mission.features[0].current_worker_run_id = child.run_id
    mission.active_worker_run_ids = [child.run_id]
    manager._save_mission(mission)

    mission = await manager.pause("cli:direct")
    assert mission is not None
    assert mission.features[0].current_worker_run_id == child.run_id

    runs.complete_subagent(
        child.run_id,
        ChildHandoff.normalize(
            {
                "status": "success",
                "return_to_orchestrator": False,
                "summary": "feature one finished after pause",
                "what_was_done": "implemented feature one",
                "what_remains": "",
                "evidence": [{"type": "test", "value": "uv run pytest tests/test_one.py -q", "note": "passes"}],
                "discovered_issues": [],
                "next_action": "complete",
            }
        ),
    )

    mission = await manager.start_or_resume_execution("cli:direct")
    assert mission is not None
    assert mission.features[0].status == "completed"
    mission_dir = tmp_path / "missions" / mission.mission_id
    assert (mission_dir / "handoffs.jsonl").read_text(encoding="utf-8").strip()


@pytest.mark.asyncio
async def test_runner_failure_does_not_auto_dispatch_all_ready_features(tmp_path: Path) -> None:
    provider = _QueuedProvider(_planning_payload(), "not json")
    runs = RunManager(tmp_path / "workspace")
    subagents = _FakeSubagents(runs)
    manager = MissionManager(
        provider=provider,
        model="test-model",
        base_dir=tmp_path / "missions",
        runs=runs,
        subagents=subagents,
    )

    await manager.start_or_update(
        scope_key="cli:direct",
        channel="cli",
        chat_id="direct",
        message_thread_id=None,
        workspace=tmp_path / "workspace",
        goal="Build mission support",
        sender_id="u1",
    )
    await manager.approve("cli:direct")
    mission = await manager.start_or_resume_execution("cli:direct")

    assert mission is not None
    assert mission.status == "paused"
    assert not subagents.spawn_calls


@pytest.mark.asyncio
async def test_mission_worker_runtime_blocks_writes_to_canonical_mission_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mission_dir = tmp_path / "missions" / "mission-1"
    mission_dir.mkdir(parents=True)
    protected = mission_dir / "state.json"
    protected.write_text('{"status":"planning"}\n', encoding="utf-8")

    provider = _SubagentProvider(
        [
            LLMResponse(
                content="write protected file",
                tool_calls=[
                    ToolCallRequest(
                        id="w1",
                        name="write_file",
                        arguments={"path": str(protected), "content": '{"status":"mutated"}\n'},
                    )
                ],
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "status": "success",
                        "return_to_orchestrator": False,
                        "summary": "done",
                        "what_was_done": "attempted protected write",
                        "what_remains": "",
                        "evidence": [],
                        "discovered_issues": [],
                        "next_action": "",
                    }
                )
            ),
        ]
    )
    subagents = SubagentManager(provider=provider, workspace=workspace, bus=MessageBus(), model="test-model")

    await subagents._run_subagent(
        "adhoc-1",
        "{}",
        "feat-001",
        {"channel": "cli", "chat_id": "direct"},
        system_prompt="mission worker",
        protected_paths=[str(protected)],
    )

    assert protected.read_text(encoding="utf-8") == '{"status":"planning"}\n'


@pytest.mark.asyncio
async def test_validation_contract_coverage_fails_for_unclaimed_assertion(tmp_path: Path) -> None:
    provider = _QueuedProvider(
        json.dumps(
            {
                "mission_title": "Broken Mission",
                "mission_summary": "Broken summary",
                "features": [
                    {
                        "id": "feat-001",
                        "description": "Only feature",
                        "status": "pending",
                        "milestone": "ms-001",
                        "preconditions": [],
                        "expected_behavior": ["Works"],
                        "verification_steps": ["Run tests"],
                        "fulfills": ["va-001"],
                        "skill_name": "",
                    }
                ],
                "milestones": [{"id": "ms-001", "title": "Plan", "description": "Initial milestone"}],
                "validation_assertions": [
                    {"id": "va-001", "assertion": "Covered", "rationale": ""},
                    {"id": "va-002", "assertion": "Unclaimed", "rationale": ""},
                ],
            }
        )
    )
    manager = MissionManager(provider=provider, model="test-model", base_dir=tmp_path)

    with pytest.raises(ValueError, match="unclaimed assertions"):
        await manager.start_or_update(
            scope_key="cli:direct",
            channel="cli",
            chat_id="direct",
            message_thread_id=None,
            workspace=tmp_path / "workspace",
            goal="Build mission support",
            sender_id="u1",
        )


def test_orchestrator_prompt_encodes_role_separation(tmp_path: Path) -> None:
    prompt = MissionPromptBuilder.build_orchestrator_prompt(tmp_path)

    assert "do not implement feature work directly" in prompt.lower()
    assert "capture every user requirement" in prompt.lower()
    assert "validation assertions" in prompt.lower()


def test_worker_prompt_encodes_handoff_requirement(tmp_path: Path) -> None:
    prompt = MissionPromptBuilder.build_worker_prompt(
        tmp_path,
        tmp_path / "mission",
        "feat-001",
        "Do the work",
    )

    assert "assigned to a single mission feature" in prompt.lower()
    assert "return exactly one json handoff" in prompt.lower()
    assert "do not mutate canonical mission files" in prompt.lower()
