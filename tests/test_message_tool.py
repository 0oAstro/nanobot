import pytest

from nanobot.agent.tools.message import MessageTool


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_allows_media_without_text() -> None:
    sent = []

    async def _send(msg) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=_send, default_channel="telegram", default_chat_id="123")
    result = await tool.execute(media=["/tmp/demo.png"])

    assert result == "Sent 1 attachment(s) to telegram:123"
    assert sent[0].media == ["/tmp/demo.png"]


@pytest.mark.asyncio
async def test_message_tool_rejects_empty_text_and_media() -> None:
    async def _send(_msg) -> None:
        return None

    tool = MessageTool(
        send_callback=_send,
        default_channel="telegram",
        default_chat_id="123",
    )

    result = await tool.execute()

    assert result == "Error: No content or media to send"
