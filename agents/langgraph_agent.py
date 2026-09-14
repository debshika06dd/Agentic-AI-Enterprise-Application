from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from tools.calculator import calculator
from tools.knowledge_search import knowledge_search


load_dotenv()


# -----------------------------
# 1. Create the Gemini model
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# 2. Define our tools
# -----------------------------

tools = [
    calculator,
    knowledge_search
]


# -----------------------------
# 3. Give tools to Gemini
# -----------------------------

llm_with_tools = llm.bind_tools(tools)


# -----------------------------
# 4. Create the LLM node
# -----------------------------

def call_model(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# -----------------------------
# 5. Create the graph
# -----------------------------

builder = StateGraph(MessagesState)


# Add LLM node
builder.add_node(
    "llm",
    call_model
)


# Add ToolNode
builder.add_node(
    "tools",
    ToolNode(tools)
)


# -----------------------------
# 6. Define graph flow
# -----------------------------

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


# -----------------------------
# 7. Compile the graph
# -----------------------------

graph = builder.compile()


# -----------------------------
# 8. Ask the agent a question
# -----------------------------

question = "How many casual leaves do employees receive?"


result = graph.invoke({
    "messages": [
        {
            "role": "user",
            "content": question
        }
    ]
})


# -----------------------------
# 9. Print final response
# -----------------------------

print("\nFinal Answer:\n")

print(
    result["messages"][-1].content
)