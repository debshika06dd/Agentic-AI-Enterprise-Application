from langchain_core.tools import tool

from rag.retriever import get_retriever


retriever = get_retriever()


@tool
def knowledge_search(query: str) -> str:
    """
    Search the company's internal knowledge base.

    Use this tool when the user asks about company policies,
    employee information, work from home, leave policies,
    security policies, or other internal company knowledge.
    """

    results = retriever.invoke(query)

    if not results:
        return "No relevant information was found."

    output = []

    for document in results:

        output.append(
            f"Source: {document.metadata.get('source')}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(output)