# ReAct Agent Exploration

An exploration of the **ReAct (Reason + Act)** framework using LangGraph, with a RAG pipeline as the working example.

## Overview

This project implements a ReAct-style agent loop where the LLM iteratively observes context, reasons about what information it needs, acts by calling tools, and repeats until it can produce a final answer. The Cisco 360 Partner Program PDFs serve as the knowledge base to exercise the pattern against.

## How the ReAct Loop Works

1. **Observe** — The agent receives the conversation history (user question + any prior tool results)
2. **Reason** — The LLM decides whether it has enough context to answer or needs to call a tool
3. **Act** — If a tool call is emitted, LangGraph routes to the `ToolNode`, executes it, and feeds results back to the agent
4. **Repeat** — The loop continues until the LLM responds without tool calls, at which point the graph exits

This is wired as a `StateGraph` in `main.py` with a conditional edge (`should_continue`) that inspects the last message for tool calls.

## Project Structure

| File | Role |
|------|------|
| `main.py` | StateGraph definition, agent node, conditional routing, streaming CLI |
| `tools.py` | `@tool`-decorated functions the agent can invoke |
| `embed.py` | Document ingestion, ChromaDB vector store, BM25 index |
| `grep.py` | Async regex search across PDF text |
| `provider.py` | promptfoo custom provider for automated evaluation |
| `promptfooconfig.yaml` | Evaluation test suite |

## Tools Available to the Agent

- **`vector_search`** — Semantic similarity via ChromaDB embeddings
- **`bm25_search`** — Keyword relevance via BM25
- **`grep_search`** — Regex pattern matching across raw PDF pages

The agent decides which tools to call (and how many times) based on the query — this is the core of the ReAct pattern.
