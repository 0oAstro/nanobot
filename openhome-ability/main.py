import httpx
from src.agent.capability import MatchingCapability
from src.main import AgentWorker
from src.agent.capability_worker import CapabilityWorker

NANOBOT_API_URL = "https://hestia.kitty-armadillo.ts.net/chat"
NANOBOT_TIMEOUT = 120.0


class NanobotCapability(MatchingCapability):
    worker: AgentWorker = None
    capability_worker: CapabilityWorker = None

    # Do not change following tag of register capability
    #{{register capability}}

    async def first_function(self):
        user_inquiry = await self.capability_worker.wait_for_complete_transcription()

        await self.capability_worker.speak("Let me think about that.")

        try:
            async with httpx.AsyncClient(timeout=NANOBOT_TIMEOUT) as client:
                resp = await client.post(
                    NANOBOT_API_URL,
                    json={"message": user_inquiry, "session": "openhome:voice"},
                )
                resp.raise_for_status()
                data = resp.json()
                raw_response = data.get("response", "")
        except Exception as e:
            raw_response = f"Sorry, I couldn't reach nanobot: {e}"

        # Condense for voice output
        voice_response = self.capability_worker.text_to_text_response(
            f"Rewrite this for spoken voice output. Be concise and natural. "
            f"Remove any markdown, code blocks, or formatting:\n\n{raw_response}",
            [],
            "You convert text into natural spoken language. Keep it brief.",
        )

        self.worker.editor_logging_handler.info(raw_response)
        await self.capability_worker.speak(voice_response)
        self.capability_worker.resume_normal_flow()

    def call(self, worker: AgentWorker):
        self.worker = worker
        self.capability_worker = CapabilityWorker(self)
        self.worker.session_tasks.create(self.first_function())
