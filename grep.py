import re
from langchain_community.document_loaders import PyPDFLoader


async def grep_pdfs(pattern: str, pdf_paths: list[str]) -> list[dict]:
    """Search PDF files for lines matching a regex pattern.

    Args:
        pattern: A regex pattern string.
        pdf_paths: List of PDF file paths to search.

    Returns:
        List of dicts with 'file', 'page', 'line_number', and 'line' for each match.
    """
    regex = re.compile(pattern)
    results = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = await loader.aload()
        for page in pages:
            page_num = page.metadata.get("page", 0)
            for line_num, line in enumerate(page.page_content.splitlines(), start=1):
                if regex.search(line):
                    results.append({
                        "file": path,
                        "page": page_num,
                        "line_number": line_num,
                        "line": line,
                    })

    return results
