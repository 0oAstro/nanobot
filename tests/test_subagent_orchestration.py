from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.runs import ChildHandoff


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")


@pytest.mark.asyncio
async def test_parent_wait_suspends_and_marks_run_waiting(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content="Waiting",
            tool_calls=[ToolCallRequest(id="w1", name="wait_for_subagents", arguments={})],
        )
    )

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="do work")
    result = await loop._process_message(msg)

    assert result is None
    runs = sorted((tmp_path / "runs").glob("*.json"))
    assert len(runs) == 1
    payload = json.loads(runs[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "parent"
    assert payload["status"] == "waiting"
    assert payload["pending_messages"]


@pytest.mark.asyncio
async def test_resume_parent_run_uses_batched_handoffs_only(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    wait_then_resume = [
        LLMResponse(
            content="Waiting",
            tool_calls=[ToolCallRequest(id="w1", name="wait_for_subagents", arguments={})],
        ),
        LLMResponse(content="done", tool_calls=[]),
    ]
    captured_second_messages: list[dict] = []

    async def fake_chat_with_retry(*, messages, **kwargs):
        if len(wait_then_resume) == 1:
            captured_second_messages[:] = messages
        return wait_then_resume.pop(0)

    loop.provider.chat_with_retry = fake_chat_with_retry

    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="do work")
    await loop._process_message(msg)

    parent = next(iter(loop.runs.runs_dir.glob("*.json")))
    parent_id = parent.stem
    child = loop.runs.create_subagent_run(
        parent_run_id=parent_id,
        session_key="cli:direct",
        goal="child task",
        label="child",
    )
    loop.runs.complete_subagent(
        child.run_id,
        ChildHandoff.normalize(
            {
                "status": "success",
                "return_to_orchestrator": False,
                "summary": "child finished",
                "what_was_done": "explored code",
                "what_remains": "",
                "evidence": [{"type": "observation", "value": "ok", "note": ""}],
                "discovered_issues": [],
                "next_action": "continue",
            }
        ),
    )

    out = await loop._resume_parent_run(
        parent_run_id=parent_id,
        channel_hint="cli",
        chat_id_hint="direct",
    )

    assert out is not None
    assert out.content == "done"
    user_messages = [m for m in captured_second_messages if m.get("role") == "user"]
    assert user_messages
    assert "Subagent handoff batch" in user_messages[-1]["content"]
    assert "transcript" not in user_messages[-1]["content"].lower()


@pytest.mark.asyncio
async def test_subagent_completion_queues_single_batched_wake(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    parent = loop.runs.create_parent_run(
        session_key="cli:direct",
        goal="goal",
        messages=[{"role": "system", "content": "sys"}],
    )
    loop.runs.mark_waiting(parent.run_id, parent.pending_messages, [])

    child1 = loop.runs.create_subagent_run(
        parent_run_id=parent.run_id,
        session_key="cli:direct",
        goal="a",
        label="a",
    )
    child2 = loop.runs.create_subagent_run(
        parent_run_id=parent.run_id,
        session_key="cli:direct",
        goal="b",
        label="b",
    )

    await loop.subagents._announce_result(
        child1.run_id,
        ChildHandoff.normalize({"status": "success", "summary": "a", "what_was_done": "a", "what_remains": ""}),
        {"channel": "cli", "chat_id": "direct"},
    )
    await loop.subagents._announce_result(
        child2.run_id,
        ChildHandoff.normalize({"status": "success", "summary": "b", "what_was_done": "b", "what_remains": ""}),
        {"channel": "cli", "chat_id": "direct"},
    )

    first = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert first.metadata["resume_parent_run_id"] == parent.run_id
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_subagent_can_complete_via_return_handoff_tool(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="handoff-1",
                    name="return_handoff",
                    arguments={
                        "status": "success",
                        "return_to_orchestrator": False,
                        "summary": "date retrieved",
                        "what_was_done": "ran date",
                        "what_remains": "",
                        "evidence": [
                            {
                                "type": "command",
                                "value": "date",
                                "note": "returned current UTC time",
                            }
                        ],
                        "discovered_issues": [],
                        "next_action": "continue",
                    },
                )
            ],
        )
    )
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")

    parent = loop.runs.create_parent_run(
        session_key="cli:direct",
        goal="goal",
        messages=[{"role": "system", "content": "sys"}],
    )
    loop.runs.mark_waiting(parent.run_id, parent.pending_messages, [])

    spawned = await loop.subagents.spawn(
        task="return a handoff",
        label="handoff",
        origin_channel="cli",
        origin_chat_id="direct",
        session_key="cli:direct",
        parent_run_id=parent.run_id,
    )
    run_id = spawned["run_id"]
    await loop.subagents._running_tasks[run_id]

    child = loop.runs.load(run_id)
    assert child is not None
    assert child.result_handoff is not None
    assert child.result_handoff["summary"] == "date retrieved"
    assert child.result_handoff["what_was_done"] == "ran date"


@pytest.mark.asyncio
async def test_subagent_compaction_uses_subagent_provider(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "sub-model"
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="compressed checkpoint", tool_calls=[])
    )

    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=bus, model="sub-model")

    summary = await manager.compact_history(
        "existing checkpoint",
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply"}],
    )

    assert summary == "compressed checkpoint"
    provider.chat_with_retry.assert_awaited_once()
    assert provider.chat_with_retry.await_args.kwargs["model"] == "sub-model"


@pytest.mark.asyncio
async def test_loop_routes_compaction_through_subagents(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    loop.compactor.should_compact = MagicMock(return_value=True)
    loop.compactor.compact_history = AsyncMock(side_effect=AssertionError("main compactor used"))
    loop.subagents.compact_history = AsyncMock(return_value="subagent checkpoint")
    loop.provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="final answer", tool_calls=[])
    )

    session = loop.sessions.get_or_create("cli:direct")
    session.messages.extend(
        [
            {"role": "user", "content": "older context"},
            {"role": "assistant", "content": "older reply"},
        ]
    )

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="new turn")
    )

    assert response is not None
    assert response.content == "final answer"
    loop.subagents.compact_history.assert_awaited_once()
    assert session.summary == "subagent checkpoint"
