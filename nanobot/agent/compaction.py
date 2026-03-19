"""Simple thread checkpoint compaction."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import estimate_prompt_tokens_chain


COMPACTION_PROMPT = """You are compressing a conversation thread into a checkpoint for the same assistant.

Write a concise checkpoint that preserves:
- user preferences and standing instructions
- relevant repository or workspace context
- important decisions already made
- unfinished work and constraints

Do not include filler. Do not mention this compaction process."""


class ContextCompactor:
    """Compact session history into one checkpoint when prompt usage gets too large."""

    def __init__(self, provider: LLMProvider, model: str, threshold_pct: float = 0.9):
        self.provider = provider
        self.model = model
        self.threshold_pct = threshold_pct

    def estimate_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[int, str]:
        return estimate_prompt_tokens_chain(self.provider, self.model, messages, tools)

    def should_compact(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        context_window_tokens: int,
    ) -> bool:
        if context_window_tokens <= 0:
            return False
        threshold = max(1, int(context_window_tokens * self.threshold_pct))
        estimated, source = self.estimate_tokens(messages, tools)
        if estimated >= threshold:
            logger.info(
                "Checkpoint compaction triggered at {}/{} via {}",
                estimated,
                context_window_tokens,
                source,
            )
            return True
        return False

    async def compact_history(
        self,
        existing_summary: str | None,
        history: list[dict[str, Any]],
    ) -> str | None:
        """Return a new checkpoint summary for the session history."""
        if not history:
            return existing_summary

        parts: list[str] = []
        if existing_summary:
            parts.append(f"Existing checkpoint:\n{existing_summary}")

        formatted: list[str] = []
        for message in history:
            role = str(message.get("role") or "unknown").upper()
            content = message.get("content")
            if isinstance(content, str):
                body = content
            else:
                body = str(content)
            if body:
                formatted.append(f"{role}: {body}")
        parts.append("Conversation history:\n" + "\n\n".join(formatted))

        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": COMPACTION_PROMPT},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            model=self.model,
            tools=None,
        )
        summary = (response.content or "").strip()
        return summary or existing_summary
