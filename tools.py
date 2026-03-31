"""This module provides tools to the LLM to search the knowledge base
"""
from typing import Any, Callable, List

from langchain_core.tools import tool

from embed import myembed
from grep import grep_pdfs

# Initialize the embedding/vector store once
_embed = myembed()
_embed.rank25()

PDF_PATHS = [
    "360-partner-program-partner-value-index-cisco-partner-incentive-metrics-guide.pdf",
    "cisco-360-program-partner-faq.pdf",
]


@tool
async def vector_search(query: str, k: int = 4) -> str:
    """Search the Cisco 360 knowledge base using semantic similarity. Use this for broad or conceptual questions."""
    docs = await _embed.search_vstore(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


@tool
async def bm25_search(query: str, k: int = 4) -> str:
    """Search the Cisco 360 knowledge base using keyword matching (BM25). Use this for specific terms or exact phrases."""
    docs = await _embed.search_bm25(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


@tool
async def grep_search(pattern: str) -> str:
    """Search the Cisco 360 PDFs for lines matching an exact regex pattern. Use this when looking for specific text, numbers, or formatting."""
    matches = await grep_pdfs(pattern, PDF_PATHS)
    if not matches:
        return "No matches found."
    return "\n".join(
        f"[{m['file']} p{m['page']} L{m['line_number']}] {m['line']}"
        for m in matches
    )


TOOLS: List[Callable[..., Any]] = [vector_search, bm25_search, grep_search]