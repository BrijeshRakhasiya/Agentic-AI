import os
import math
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq  # ✅

# Load .env file
load_dotenv()

# Get configs from environment
GROQ_API_KEY="PUT_API_KEY_HERE"
GROQ_MODEL = "llama3-8b-8192"

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.3
)


# Define tools
wiki = WikipediaAPIWrapper()

def safe_calculator(expression):
    """Safe calculator that supports basic math operations and functions"""
    # Create a safe namespace with math functions
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

tools = [
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Looks up topics and people using Wikipedia"
    ),
    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Performs math operations. Supports +, -, *, /, sqrt(), pow(), etc. Example: sqrt(1972)"
    )
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
query = "Who is Sundar Pichai and what is the square root of his birth year?"
response = agent.invoke({"input": query})
print("\n🧠 Final Answer:\n", response["output"])
