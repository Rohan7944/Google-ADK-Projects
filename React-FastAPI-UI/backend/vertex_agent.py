import requests
from google.auth import default
from google.auth.transport.requests import Request


def query_agent_engine(
    *,
    user_message: str,
    agent_id: str,
    project_id: str,
    location: str,
) -> str:
    """
    Calls Vertex AI Agent Engine using ADC
    """

    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())

    access_token = credentials.token

    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/"
        f"agentEngines/{agent_id}:generateContent"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}],
            }
        ]
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Typical response shape from Agent Engines
    return (
        data["candidates"][0]["content"]["parts"][0]["text"]
        if "candidates" in data
        else "No response from agent."
    )