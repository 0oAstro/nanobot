from types import SimpleNamespace

import pytest

from nanobot.agent.tools.cron import CronTool
from nanobot.cron.types import CronSchedule


class _FakeCronService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_job(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(name=kwargs["name"], id="job-1")


def test_cron_tool_add_job_preserves_tz() -> None:
    cron = _FakeCronService()
    tool = CronTool(cron)
    tool.set_context("telegram", "123")

    result = tool._add_job(
        "Morning reminder",
        every_seconds=None,
        cron_expr="0 7 * * *",
        tz=None,
        at=None,
    )

    assert result == "Created job 'Morning reminder' (id: job-1)"
    assert cron.calls[0]["schedule"] == CronSchedule(
        kind="cron",
        expr="0 7 * * *",
        tz=None,
    )


@pytest.mark.asyncio
async def test_cron_tool_execute_maps_timezone_alias_to_tz() -> None:
    cron = _FakeCronService()
    tool = CronTool(cron)
    tool.set_context("telegram", "123")

    result = await tool.execute(
        action="add",
        message="Morning reminder",
        cron_expr="0 7 * * *",
        timezone="Asia/Ho_Chi_Minh",
    )

    assert result == "Created job 'Morning reminder' (id: job-1)"
    assert cron.calls[0]["schedule"] == CronSchedule(
        kind="cron",
        expr="0 7 * * *",
        tz="Asia/Ho_Chi_Minh",
    )
