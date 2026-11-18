"""Run the scoped memory example with a synthetic preference."""

import os

from pydantic import BaseModel

import instructor
from examples.persistent_memory_dakera.dakera_memory import create_with_memory


class Reply(BaseModel):
    answer: str


def main() -> None:
    from dakera import DakeraClient

    model = instructor.from_provider("openai/gpt-6-luna")
    with DakeraClient(
        base_url=os.environ.get("DAKERA_BASE_URL", "http://localhost:3300"),
        api_key=os.environ["DAKERA_API_KEY"],
    ) as memory:
        reply = create_with_memory(
            model,
            memory,
            Reply,
            agent_id=os.environ["DAKERA_AGENT_ID"],
            message="I prefer dark mode. Which display setting should I use?",
            remember_quote="I prefer dark mode.",
        )
        print(reply.answer)


if __name__ == "__main__":
    main()
