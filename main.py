import asyncio
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep

from tools import TOOLS

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
from typing_extensions import Annotated


model = ChatOllama(
    base_url="http://192.168.1.17:11434/",
    model="qwen3.5",
)

@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


if __name__ == '__main__':
  try:
    main()
    print("HI")
  except KeyboardInterrupt:
    print('\nGoodbye!')

