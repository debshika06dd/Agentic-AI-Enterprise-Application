from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from prompts.agent_prompt import agent_prompt

# Load environment variables
load_dotenv()

# Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Connect prompt and LLM
chain = agent_prompt | llm

# User question
question = "Explain what an AI agent is in simple words."

# Invoke the chain
response = chain.invoke({
    "question": question
})

# Print response
print(response.content)