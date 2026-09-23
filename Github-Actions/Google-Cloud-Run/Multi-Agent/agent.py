"""
Core Agent Configuration and Initialization.
"""
import os
from google.adk.agents import Agent
from .tools import fetch_system_status, calculate_projected_growth

# Enforce explicit checking for required API configurations during execution
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Missing runtime dependency: GOOGLE_API_KEY environment variable.")

# The deployment tool strictly hooks onto the 'root_agent' object variable
root_agent = Agent(
    name="multi_tool_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a production operations assistant hosted on Google Cloud Run. "
        "You have direct access to internal server tool suites to look up system metrics "
        "and calculate business growth trends for operators. Be clear and succinct."
    ),
    # Supply your imported Python functions natively as tools
    tools=[
        fetch_system_status, 
        calculate_projected_growth
    ]
)
