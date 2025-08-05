import os
import math
from langchain_groq import ChatGroq
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper

# === Config ===
GROQ_API_KEY="PUT_API_KEY_HERE"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192", 
    temperature=0.3
)

# === Wikipedia Expert Agent ===
wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Looks up people and topics on Wikipedia"
)
wiki_agent = initialize_agent(
    tools=[wiki_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === Math Expert Agent ===
def safe_calculator(expression):
    """Safe calculator that supports basic math operations"""
    safe_dict = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "exp": math.exp,
        "log": math.log, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi, "e": math.e
    }
    try:
        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

calc_tool = Tool(
    name="Calculator",
    func=safe_calculator,
    description="Performs basic math operations. Use ** 0.5 for square root"
)
math_agent = initialize_agent(
    tools=[calc_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === Coordinator Agent Logic ===
def multi_agent_coordinator(task: str) -> str:
    print("\n🤖 Step 1: Ask Wikipedia Agent...")
    wiki_response = wiki_agent.invoke({"input": "Summarize Sundar Pichai's background and give his birth year"})
    wiki_summary = wiki_response["output"]
    
    print("\n🤖 Step 2: Ask Math Agent...")
    year = "1972" if "1972" in wiki_summary else input("Enter the birth year you found: ")
    math_response = math_agent.invoke({"input": f"{year} ** 0.5"})
    sqrt = math_response["output"]
    
    return f"{wiki_summary}\n\nThe square root of {year} is {sqrt.strip()}."

# === Run
if __name__ == "__main__":
    final = multi_agent_coordinator("Summarize Sundar Pichai and do math")
    print("\n✅ Final Response:\n", final)
