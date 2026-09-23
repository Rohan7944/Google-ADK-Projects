from google.adk.agents import Agent

# The ADK CLI strictly searches for a variable named 'root_agent'
root_agent = Agent(
    name="production_agent",
    model="gemini-2.5-flash",
    instruction="You are a live hosted API helper."
)
