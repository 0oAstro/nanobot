"""Minimal HTTP API channel for nanobot.

Exposes a single POST /chat endpoint that accepts a JSON body
{"message": "...", "session": "openhome:voice"} and returns
the agent's response as {"response": "..."}.

Started automatically by the gateway when enabled in config,
or run standalone via `python -m nanobot.channels.http_api`.
"""

from aiohttp import web
from loguru import logger


def create_app(agent_loop):
    """Create an aiohttp app wired to the given AgentLoop."""

    async def handle_chat(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        message = body.get("message", "").strip()
        if not message:
            return web.json_response({"error": "message is required"}, status=400)

        session = body.get("session", "http:api")

        logger.info(f"HTTP API request: session={session} message={message[:80]}...")

        try:
            response = await agent_loop.process_direct(
                message,
                session_key=session,
                channel="http",
                chat_id="api",
            )
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"response": response})

    async def handle_health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/chat", handle_chat)
    app.router.add_get("/health", handle_health)
    return app


async def start_http_api(agent_loop, *, host: str = "0.0.0.0", port: int = 8318):
    """Start the HTTP API server (non-blocking, returns the runner)."""
    app = create_app(agent_loop)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"HTTP API listening on {host}:{port}")
    return runner
