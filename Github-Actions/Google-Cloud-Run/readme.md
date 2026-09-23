The workflow can be fully automated to deploy your Google ADK agent directly to Google Cloud Run whenever changes are merged into your main branch.

Because the Google ADK CLI includes a built-in orchestration command (adk deploy cloud_run), configuring this inside GitHub Actions is highly optimized.

Prerequisites for Google Cloud

To transition away from hardcoded or long-lived API keys, use Workload Identity Federation (WIF):

1. Create a Service Account in your Google Cloud Project with the following roles: Cloud Run Admin, Artifact Registry Admin, Storage Admin, Cloud Build Editor, and Service Account User.
2. Set up a Workload Identity Pool to trust your GitHub repository.
3. Save your GCP PROJECT_ID, WIF_PROVIDER (the full provider path string), and WIF_SERVICE_ACCOUNT address as GitHub Actions Secrets or environment variables.

A. The Deployment GitHub Actions Workflow: 

Create or replace .github/workflows/deploy-adk.yml. This script specifically identifies the pull request merge event by using a conditional filter (if: github.event.pull_request.merged == true) alongside direct pushes to main.

B. Valid Directory Structure Required by ADK

To deploy your agent seamlessly using the `adk deploy cloud_run` workflow, your repository must follow this specific structural blueprint. The ADK deployment tool expects a target module directory containing an `__init__.py` file that explicitly exposes your configured agent.

```text
.
├── .github/
│   └── workflows/
│       └── deploy-adk.yml     # The GitHub Actions workflow file
├── requirements.txt           # Main project dependencies (must include google-adk)
└── src/
    └── my_adk_agent/          # Root directory for your agent module
        ├── __init__.py        # Exposes the agent to the ADK buildpack
        └── agent.py           # Contains the core Agent instantiation logic
```

How it Works Behind the Scenes

1. github.event.pull_request.merged == true: This validation step guarantees your pipeline will not deploy broken code if a user closes a Pull Request by discarding it without merging.
   
2. Automated Containment: The command adk deploy cloud_run reads your Python logic, compiles a temporary Docker configuration container, builds it via Google Cloud Build, and hosts it automatically on a scale-to-zero serverless endpoint.

3. --with-ui and --trace-to-cloud: Automatically exposes a developer interface playground URL and syncs diagnostic traces to Google Cloud Observability.
