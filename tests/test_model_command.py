"""Tests for /model, /reload, and slash command aliases."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.session.manager import Session


def _make_loop(tmp_path, *, model: str = "main-model"):
    """Create a minimal AgentLoop for command testing."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = model
    provider.list_models = AsyncMock(return_value=[("z-model", "z-model"), ("a-model", "a-model")])

    session = Session(key="telegram:123")

    with (
        patch.object(AgentLoop, "_register_default_tools", lambda self: None),
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SubagentManager") as mock_subagents,
    ):
        subagents = MagicMock()
        mock_subagents.return_value = subagents
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model=model)

    loop.sessions = MagicMock()
    loop.sessions.get_or_create.return_value = session
    loop.memory_consolidator = MagicMock()
    loop.memory_consolidator.archive_messages = AsyncMock(return_value=True)
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)
    return loop, provider, subagents


@pytest.mark.asyncio
async def test_loop_uses_explicit_subagent_model_when_provided(tmp_path):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"

    with (
        patch.object(AgentLoop, "_register_default_tools", lambda self: None),
        patch("nanobot.agent.loop.ContextBuilder"),
        patch("nanobot.agent.loop.SubagentManager") as mock_subagents,
    ):
        AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            model="main-model",
            subagent_model="sub-model",
        )

    assert mock_subagents.call_args.kwargs["model"] == "sub-model"


@pytest.mark.asyncio
async def test_model_status_shows_main_and_following_subagent(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    config = Config()
    config.agents.defaults.model = "main-model"
    config.agents.defaults.subagent_model = None

    with patch("nanobot.agent.loop.load_config", return_value=config):
        response = await loop._process_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="123", content="/model")
        )

    assert response is not None
    assert response.content == "Main: main-model\nSubagent: main-model (follows main)"
    assert response.metadata["_model_list"] == ["a-model", "z-model"]
    assert response.metadata["_current_model"] == "main-model"


@pytest.mark.asyncio
async def test_model_status_prefers_configured_provider_catalog(tmp_path):
    loop, provider, _subagents = _make_loop(tmp_path)
    config = Config()
    config.agents.defaults.model = "main-model"
    config.agents.defaults.provider = "anthropic"
    config.providers.anthropic.models = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

    with patch("nanobot.agent.loop.load_config", return_value=config):
        response = await loop._process_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="123", content="/model")
        )

    assert response is not None
    assert response.metadata["_model_list"] == [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
    ]
    provider.list_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_subagent_command_persists_explicit_subagent_model(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    config = Config()
    config.agents.defaults.model = "main-model"

    loop._reload_runtime_from_config = AsyncMock(return_value=("main-model", "sub-model"))

    with (
        patch("nanobot.agent.loop.load_config", return_value=config),
        patch("nanobot.agent.loop.save_config") as mock_save,
    ):
        response = await loop._process_message(
            InboundMessage(
                channel="telegram",
                sender_id="u1",
                chat_id="123",
                content="/model subagent sub-model",
            )
        )

    assert config.agents.defaults.subagent_model == "sub-model"
    mock_save.assert_called_once_with(config)
    loop._reload_runtime_from_config.assert_awaited_once_with(config)
    assert response is not None
    assert "Subagent model updated." in response.content
    assert "Subagent model: sub-model" in response.content


@pytest.mark.asyncio
async def test_model_subagent_clear_reverts_to_main_model(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    config = Config()
    config.agents.defaults.model = "main-model"
    config.agents.defaults.subagent_model = "old-sub-model"

    loop._reload_runtime_from_config = AsyncMock(return_value=("main-model", "main-model"))

    with (
        patch("nanobot.agent.loop.load_config", return_value=config),
        patch("nanobot.agent.loop.save_config") as mock_save,
    ):
        response = await loop._process_message(
            InboundMessage(
                channel="telegram",
                sender_id="u1",
                chat_id="123",
                content="/model subagent clear",
            )
        )

    assert config.agents.defaults.subagent_model is None
    mock_save.assert_called_once_with(config)
    loop._reload_runtime_from_config.assert_awaited_once_with(config)
    assert response is not None
    assert "Subagent model: main-model" in response.content


@pytest.mark.asyncio
async def test_reload_reloads_runtime_in_process(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop._reload_runtime_from_config = AsyncMock(return_value=("main-model", "sub-model"))

    response = await loop._process_message(
        InboundMessage(channel="telegram", sender_id="u1", chat_id="123", content="/reload")
    )

    loop._reload_runtime_from_config.assert_awaited_once_with()
    assert response is not None
    assert response.content == (
        "Reloaded config/runtime successfully.\nMain model: main-model\nSubagent model: sub-model"
    )


@pytest.mark.asyncio
async def test_help_command_accepts_telegram_command_suffix(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)

    response = await loop._process_message(
        InboundMessage(
            channel="telegram", sender_id="u1", chat_id="123", content="/help@nanobot_test"
        )
    )

    assert response is not None
    assert "/model — Pick or set main/subagent model" in response.content


@pytest.mark.asyncio
async def test_new_command_accepts_telegram_command_suffix(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.sessions.get_or_create.return_value.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    response = await loop._process_message(
        InboundMessage(
            channel="telegram", sender_id="u1", chat_id="123", content="/new@nanobot_test"
        )
    )

    loop.memory_consolidator.archive_messages.assert_called_once_with(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    loop.sessions.save.assert_called_once()
    loop.sessions.invalidate.assert_called_once()
    assert response is not None
    assert response.content == "New session started."


@pytest.mark.asyncio
async def test_mission_help_command_accepts_telegram_command_suffix(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()
    loop.missions.get_active_mission.return_value = None
    loop.missions.mission_help_text.return_value = "mission help"

    response = await loop._process_message(
        InboundMessage(
            channel="telegram", sender_id="u1", chat_id="123", content="/mission@nanobot_test"
        )
    )

    assert response is not None
    assert response.content == "mission help"


@pytest.mark.asyncio
async def test_mission_command_starts_new_mission(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()
    mission = MagicMock()
    loop.missions.start_or_update = AsyncMock(return_value=mission)
    loop.missions.format_summary.return_value = "Mission: Test"

    response = await loop._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="/mission build a mission layer",
        )
    )

    loop.missions.start_or_update.assert_awaited_once_with(
        scope_key="telegram:123",
        channel="telegram",
        chat_id="123",
        message_thread_id=None,
        workspace=tmp_path,
        goal="build a mission layer",
        sender_id="u1",
    )
    assert response is not None
    assert response.content == "Mission: Test"


@pytest.mark.asyncio
async def test_mission_command_approve_calls_mission_manager(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()
    mission = MagicMock()
    loop.missions.approve = AsyncMock(return_value=mission)
    loop.missions.runs = MagicMock()
    loop.missions.subagents = MagicMock()
    loop.missions.start_or_resume_execution = AsyncMock(return_value=mission)
    loop.missions.format_summary.return_value = "Mission: Approved"

    response = await loop._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="/mission approve",
        )
    )

    loop.missions.approve.assert_awaited_once_with("telegram:123")
    loop.missions.start_or_resume_execution.assert_awaited_once_with("telegram:123")
    assert response is not None
    assert response.content == "Mission: Approved"


@pytest.mark.asyncio
async def test_mission_command_start_returns_approval_error(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()
    loop.missions.start_or_resume_execution = AsyncMock(
        side_effect=ValueError("Mission approval is required before execution can start.")
    )

    response = await loop._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="/mission start",
        )
    )

    loop.missions.start_or_resume_execution.assert_awaited_once_with("telegram:123")
    assert response is not None
    assert "approval is required" in response.content.lower()


@pytest.mark.asyncio
async def test_active_mission_plain_approved_stays_in_normal_chat(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()
    mission = MagicMock()
    mission.status = "planning"
    mission.approval_required = True
    loop.missions.get_active_mission.return_value = mission
    loop._run_agent_loop = AsyncMock(return_value=("normal reply", None, [], None))
    loop.context.build_messages.return_value = [{"role": "user", "content": "approved"}]

    response = await loop._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="approved",
        )
    )

    loop.missions.approve.assert_not_called()
    assert response is not None
    assert response.content == "normal reply"


@pytest.mark.asyncio
async def test_active_mission_allows_freeform_chat_to_use_main_loop(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()
    mission = MagicMock()
    mission.status = "planning"
    mission.approval_required = True
    loop.missions.get_active_mission.return_value = mission
    loop._run_agent_loop = AsyncMock(return_value=("main loop reply", None, [], None))
    loop.context.build_messages.return_value = [{"role": "user", "content": "do it now"}]

    response = await loop._process_message(
        InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="do it now",
        )
    )

    loop._run_agent_loop.assert_awaited_once()
    assert response is not None
    assert response.content == "main loop reply"


@pytest.mark.asyncio
async def test_heartbeat_session_does_not_inherit_target_chat_mission_lock(tmp_path):
    loop, _provider, _subagents = _make_loop(tmp_path)
    loop.missions = MagicMock()

    mission = MagicMock()
    mission.status = "planning"
    mission.approval_required = True

    def _get_active(scope_key: str):
        return mission if scope_key == "telegram:123" else None

    loop.missions.get_active_mission.side_effect = _get_active
    loop.context.build_messages.return_value = [{"role": "user", "content": "heartbeat task"}]
    loop._run_agent_loop = AsyncMock(return_value=("heartbeat ok", None, [], None))

    response = await loop.process_direct(
        "heartbeat task",
        session_key="heartbeat",
        channel="telegram",
        chat_id="123",
    )

    assert response == "heartbeat ok"
    loop._run_agent_loop.assert_awaited_once()
    loop.missions.get_active_mission.assert_not_called()
