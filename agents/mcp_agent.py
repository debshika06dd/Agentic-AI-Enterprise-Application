import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    # --------------------------------
    # 1. Create MCP client
    # --------------------------------

    client = MultiServerMCPClient(
        {
            "employee_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp/employee_server.py"],
            }
        }
    )


    # --------------------------------
    # 2. Get tools from MCP server
    # --------------------------------

    tools = await client.get_tools()


    # --------------------------------
    # 3. Create Gemini
    # --------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


    # --------------------------------
    # 4. Give MCP tools to Gemini
    # --------------------------------

    llm_with_tools = llm.bind_tools(tools)


    # --------------------------------
    # 5. Create LLM node
    # --------------------------------

    def call_model(state: MessagesState):

        response = llm_with_tools.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }


    # --------------------------------
    # 6. Create graph
    # --------------------------------

    builder = StateGraph(MessagesState)


    # Add LLM node
    builder.add_node(
        "llm",
        call_model
    )


    # Add MCP tools as ToolNode
    builder.add_node(
        "tools",
        ToolNode(tools)
    )


    # --------------------------------
    # 7. Define graph
    # --------------------------------

    builder.add_edge(
        START,
        "llm"
    )


    builder.add_conditional_edges(
        "llm",
        tools_condition
    )


    builder.add_edge(
        "tools",
        "llm"
    )


    # --------------------------------
    # 8. Compile graph
    # --------------------------------

    graph = builder.compile()


    # --------------------------------
    # 9. Ask question
    # --------------------------------

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the leave balance of EMP001?"
                }
            ]
        }
    )


    # --------------------------------
    # 10. Print answer
    # --------------------------------

    print("\nFinal Answer:\n")

    print(
        result["messages"][-1].content
    )


if __name__ == "__main__":
    asyncio.run(main())