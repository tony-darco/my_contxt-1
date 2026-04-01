from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from typing_extensions import Annotated
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import AnyMessage, AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.managed import IsLastStep

from tools import TOOLS



model = ChatOllama(
    base_url="http://192.168.1.17:11434/",
    model="qwen3.5",
).bind_tools(TOOLS)

_embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://192.168.1.17:11434/",
)

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

TOKEN_THRESHOLD = 6000
PRUNE_RATIO = 0.30


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


async def prune(state: State) -> dict:
    """If tool-retrieved content exceeds TOKEN_THRESHOLD, drop the lowest-relevance 30%."""
    tool_msgs = [m for m in state.messages if isinstance(m, ToolMessage)]
    total_tokens = sum(_estimate_tokens(get_message_text(m)) for m in tool_msgs)

    if total_tokens <= TOKEN_THRESHOLD:
        return {"messages": []}

    # Extract the original user query
    query = next(
        (get_message_text(m) for m in state.messages if isinstance(m, HumanMessage)),
        "",
    )

    # Embed query + all tool message texts in one batch
    texts = [query] + [get_message_text(m) for m in tool_msgs]
    embeds = await _embeddings.aembed_documents(texts)
    query_vec = embeds[0]

    # Score each tool message by cosine similarity to the query
    scored = sorted(
        zip(tool_msgs, embeds[1:]),
        key=lambda pair: _cosine_sim(query_vec, pair[1]),
    )

    n_to_remove = max(1, int(len(tool_msgs) * PRUNE_RATIO))
    to_remove = [m for m, _ in scored[:n_to_remove]]

    removed_call_ids = {m.tool_call_id for m in to_remove}
    removals = [RemoveMessage(id=m.id) for m in to_remove]

    # Also remove AIMessages whose tool_calls are all being pruned
    for m in state.messages:
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            call_ids = {tc["id"] for tc in m.tool_calls}
            if call_ids <= removed_call_ids:
                removals.append(RemoveMessage(id=m.id))

    return {"messages": removals}


graph_builder = StateGraph(State, input=InputState)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("prune", prune)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "prune")
graph_builder.add_edge("prune", "agent")

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

