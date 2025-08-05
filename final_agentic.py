import os
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.llms import Groq

# === Setup ===
GROQ_API_KEY="PUT_API_KEY_HERE"
llm = Groq(model="llama3-8b-8192", temperature=0.3)

# === Define tools ===
wiki = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for looking up people, places, or topics."
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Performs math calculations given a formula."
    )
]

# === Create agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === Goal: Perform multi-step reasoning
goal = "Find the CEO of OpenAI, summarize their background, and calculate the square root of their birth year."

# === Run
print("\n🎯 Agentic AI Output:\n")
result = agent.run(goal)
print("\n✅ Final Answer:\n", result)
