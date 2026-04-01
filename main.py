from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Sequence

from typing_extensions import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.managed import IsLastStep

from tools import TOOLS


# ── Model ──────────────────────────────────────────────────────────────────────

model = ChatOllama(
    base_url="http://192.168.1.17:11434/",
    model="qwen3.5",
).bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a Cisco 360 partner-program expert. "
    "Use the provided tools to search the knowledge base for evidence before answering. "
    "You may call tools multiple times to gather enough context. "
    "Once you have sufficient evidence, provide a clear, concise answer in 2-3 sentences. "
    "Be direct and avoid unnecessary elaboration."
)



def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()



@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )

@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)



async def agent(state: State) -> dict:
    """Observe the trajectory, reason about what to do, and act (call tools or respond)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state.messages)
    response = await model.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)



def should_continue(state: State) -> str:
    """If the last message has tool calls, route to tools; otherwise end."""
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if state.is_last_step:
            return END
        return "tools"
    return END


graph_builder = StateGraph(State, input=InputState)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()



async def main():
    query = input("Ask a question about Cisco 360: ")
    print()
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=query)]},
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk.content, str) and chunk.content:
                print(chunk.content, end="", flush=True)
    print()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nGoodbye!')

