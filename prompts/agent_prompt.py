from langchain_core.prompts import ChatPromptTemplate

agent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Your responsibilities:
        1. Explain technical concepts clearly.
        2. Use simple language when possible.
        3. Give examples when helpful.
        4. Do not make up facts.
        """
    ),
    (
        "human",
        "{question}"
    )
])