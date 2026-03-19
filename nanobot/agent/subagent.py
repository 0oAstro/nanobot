"""Subagent manager for background task execution."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.handoff import ReturnHandoffTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
from nanobot.prompts import load_prompt
from nanobot.providers.base import LLMProvider
from nanobot.runs import ChildHandoff, RunManager
from nanobot.utils.helpers import build_assistant_message


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        web_search_config: "WebSearchConfig | None" = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        runs: RunManager | None = None,
        max_iterations: int = 40,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.runs = runs or RunManager(workspace)
        self.max_iterations = max_iterations
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        parent_run_id: str | None = None,
        system_prompt: str | None = None,
        protected_paths: list[str] | None = None,
    ) -> dict[str, str]:
        """Spawn a subagent to execute a task in the background."""
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}
        child = self.runs.create_subagent_run(
            parent_run_id=parent_run_id or "",
            session_key=session_key or f"{origin_channel}:{origin_chat_id}",
            goal=task,
            label=display_label,
        ) if parent_run_id else None
        task_id = child.run_id if child else f"adhoc-{len(self._running_tasks) + 1}"

        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                system_prompt=system_prompt,
                protected_paths=protected_paths,
            )
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return {"run_id": task_id, "label": display_label}

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        *,
        system_prompt: str | None = None,
        protected_paths: list[str] | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
            blocked_paths = [Path(path).expanduser().resolve() for path in (protected_paths or [])]
            tools.register(
                ReadFileTool(
                    workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
                )
            )
            tools.register(
                WriteFileTool(
                    workspace=self.workspace,
                    allowed_dir=allowed_dir,
                    blocked_paths=blocked_paths,
                )
            )
            tools.register(
                EditFileTool(
                    workspace=self.workspace,
                    allowed_dir=allowed_dir,
                    blocked_paths=blocked_paths,
                )
            )
            tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                )
            )
            tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
            tools.register(WebFetchTool(proxy=self.web_proxy))
            handoff_tool = ReturnHandoffTool()
            tools.register(handoff_tool)

            system_prompt = self._build_subagent_prompt(system_prompt)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            iteration = 0
            final_result: str | None = None
            tool_handoff: dict[str, Any] | None = None

            while iteration < self.max_iterations:
                iteration += 1

                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                )

                if response.has_tool_calls:
                    tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                    messages.append(
                        build_assistant_message(
                            response.content or "",
                            tool_calls=tool_call_dicts,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        )
                    )

                    # Execute tools
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug(
                            "Subagent [{}] executing: {} with arguments: {}",
                            task_id,
                            tool_call.name,
                            args_str,
                        )
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            }
                        )
                        if tool_call.name == handoff_tool.name and handoff_tool.handoff is not None:
                            tool_handoff = handoff_tool.handoff
                            break
                    if tool_handoff is not None:
                        break
                else:
                    final_result = response.content
                    break

            if tool_handoff is not None:
                logger.debug(
                    "Subagent [{}] completed via return_handoff tool: status={} summary={}",
                    task_id,
                    tool_handoff.get("status"),
                    tool_handoff.get("summary"),
                )
                handoff = ChildHandoff.normalize(tool_handoff)
            elif final_result is None:
                handoff = ChildHandoff.normalize(
                    None,
                    fallback_error=(
                        f"Subagent reached max iterations ({self.max_iterations}) without producing a final handoff."
                    ),
                )
            else:
                parsed_handoff = self._parse_handoff(final_result)
                if parsed_handoff is None:
                    logger.warning(
                        "Subagent [{}] returned non-JSON final result; raw_final_result={!r}",
                        task_id,
                        final_result[:800],
                    )
                else:
                    logger.debug(
                        "Subagent [{}] parsed handoff: status={} return_to_orchestrator={} summary={}",
                        task_id,
                        parsed_handoff.get("status"),
                        parsed_handoff.get("return_to_orchestrator"),
                        parsed_handoff.get("summary"),
                    )
                handoff = ChildHandoff.normalize(parsed_handoff)
                if handoff.summary == "Task completed." and parsed_handoff is None:
                    logger.warning(
                        "Subagent [{}] fell back to generic handoff summary after non-JSON completion.",
                        task_id,
                    )

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, handoff, origin)

        except Exception as e:
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(
                task_id,
                ChildHandoff.normalize(None, fallback_error=f"Subagent failed: {e}"),
                origin,
            )

    async def _announce_result(
        self,
        task_id: str,
        handoff: ChildHandoff,
        origin: dict[str, str],
    ) -> None:
        """Persist child handoff and wake the parent if it is waiting."""
        child, parent = self.runs.complete_subagent(task_id, handoff)
        if not parent:
            return
        if parent.status != "waiting" or parent.wake_queued:
            return
        self.runs.set_wake_queued(parent.run_id, True)
        await self.bus.publish_inbound(
            InboundMessage(
                channel="system",
                sender_id="subagent",
                chat_id=f"{origin['channel']}:{origin['chat_id']}",
                content=f"[subagent completion batch for parent run {parent.run_id}]",
                metadata={"resume_parent_run_id": parent.run_id},
            )
        )
        logger.debug("Subagent [{}] queued parent wake for {}", task_id, parent.run_id)

    @staticmethod
    def _parse_handoff(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        if match := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL):
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _build_subagent_prompt(self, override: str | None = None) -> str:
        """Build a focused system prompt for the subagent."""
        if override:
            return override
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [
            load_prompt(
                "subagent_system.md",
                runtime_context=time_ctx,
                workspace=self.workspace,
            )
        ]

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(
                f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}"
            )

        return "\n\n".join(parts)

    async def cancel_run(self, run_id: str) -> bool:
        """Cancel one tracked subagent run by id."""
        task = self._running_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self.runs.cancel_run(run_id)
        return task is not None

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
