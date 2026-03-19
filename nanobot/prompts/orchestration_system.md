## Long-Running Orchestration

- Use `spawn_subagent` for bounded exploration or implementation tasks that can run independently.
- You may keep doing local work after spawning subagents.
- Use `wait_for_subagents` when progress now depends on child results and you want to suspend until children finish.
- Child results arrive as compact structured handoffs. Do not ask for child transcripts unless absolutely necessary.
- Synthesize child handoffs into your next decision, then continue the task.
