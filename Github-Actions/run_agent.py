import argparse
import os
from google.adk import Agent

def main():
    # Parse incoming GitHub Actions inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="Hello")
    args = parser.parse_args()

    # Double check API Key injection
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")

    # Initialize your Google ADK agent
    print("Initializing Google ADK Agent...")
    agent = Agent(
        name="ci_helper_agent",
        model="gemini-2.5-flash",  #
        instruction="You are a helpful automated task runner running inside GitHub Actions."
    )

    # Execute agent command
    print(f"Sending prompt to agent: {args.prompt}")
    response = agent.run(args.prompt)
    print("\n--- Agent Response ---")
    print(response)

if __name__ == "__main__":
    main()
