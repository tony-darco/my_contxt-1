"""Promptfoo custom Python provider that invokes the LangGraph search agent."""
from __future__ import annotations

import asyncio
from langchain_core.messages import HumanMessage
from main import graph, get_message_text


def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Entry point for promptfoo. Runs the graph and returns the final answer."""

    async def _run():
        result = await graph.ainvoke({"messages": [HumanMessage(content=prompt)]})
        return get_message_text(result["messages"][-1])

    output = asyncio.run(_run())

    return {
        "output": output,
    }
