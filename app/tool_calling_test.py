from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from tools.calculator import calculator


# Load environment variables
load_dotenv()


# Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# Register tools
tools = [calculator]

llm_with_tools = llm.bind_tools(tools)


# User question
question = "What is 125 multiplied by 48?"


# First LLM call
response = llm_with_tools.invoke(question)


# Check whether the model requested a tool
if response.tool_calls:

    print("Tool call requested:")
    print(response.tool_calls)


    # Execute each requested tool
    tool_results = []

    for tool_call in response.tool_calls:

        if tool_call["name"] == "calculator":

            result = calculator.invoke(
                tool_call["args"]
            )

            tool_results.append(
                {
                    "tool_call": tool_call,
                    "result": result
                }
            )


    # Print tool result
    print("\nTool result:")

    for item in tool_results:
        print(item["result"])

else:

    print("No tool was requested.")