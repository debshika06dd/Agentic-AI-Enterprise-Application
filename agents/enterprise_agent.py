import asyncio

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.calculator import calculator
from tools.knowledge_search import knowledge_search


load_dotenv()


async def main():

    # --------------------------------
    # 1. Connect to MCP server
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
    # 2. Get MCP tools
    # --------------------------------

    mcp_tools = await client.get_tools()


    print("MCP tools available:")

    for tool in mcp_tools:
        print("-", tool.name)


    # --------------------------------
    # 3. Combine all tools
    # --------------------------------

    tools = [
        calculator,
        knowledge_search,
        *mcp_tools
    ]


    # --------------------------------
    # 4. Create Gemini
    # --------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


    # --------------------------------
    # 5. Bind tools to Gemini
    # --------------------------------

    llm_with_tools = llm.bind_tools(tools)


    # --------------------------------
    # 6. Define LLM node
    # --------------------------------

    def call_model(state: MessagesState):

        response = llm_with_tools.invoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }


    # --------------------------------
    # 7. Build LangGraph
    # --------------------------------

    builder = StateGraph(MessagesState)


    builder.add_node(
        "llm",
        call_model
    )


    builder.add_node(
        "tools",
        ToolNode(tools)
    )


    # --------------------------------
    # 8. Define graph flow
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
    # 9. Compile graph
    # --------------------------------

    graph = builder.compile()


    # --------------------------------
    # 10. Ask the agent
    # --------------------------------

    question = "How many casual leaves do employees receive?"


    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )


    # --------------------------------
    # 11. Print final answer
    # --------------------------------

    print("\nFinal Answer:\n")

    final_content = result["messages"][-1].content

    if isinstance(final_content, list):

        for block in final_content:

            if isinstance(block, dict) and block.get("type") == "text":

                print(block.get("text"))

    else:

        print(final_content)


if __name__ == "__main__":

    asyncio.run(main())
