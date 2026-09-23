To run a sample Google Agent Development Kit (ADK) agent automatically via GitHub Actions, you need a workflow file that provisions Python, installs the google-adk package, and injects your GOOGLE_API_KEY securely using GitHub Secrets.

Here is a production-ready GitHub Actions workflow configuration.

1. The GitHub Actions Workflow File: Create a file named .github/workflows/run-adk-agent.yml in your repository:
2. Accompanying Python Script Sample (run_agent.py): Ensure your root executable Python file (e.g., run_agent.py) grabs the CLI argument or input properly and uses the environment API key

Adding Your Google API Key to GitHub

To make this work seamlessly without leaking your private credentials:

1. Go to your repository on GitHub.
2. Navigate to Settings > Secrets and variables > Actions.
3. Click New repository secret.
4. Set the Name to GOOGLE_API_KEY.
5. Set the Value to your actual key from Google AI Studio.
6. Click Add secret

Here is the file(workflows-run-adk-agent-pr.yml) configured to trigger automatically on push or pull request events, rather than just waiting for a manual trigger.
To prevent unnecessary API costs and runaway cycles, this setup targets changes on specific branches (main or developer) and monitors changes made specifically to your codebase or configuration folders.
