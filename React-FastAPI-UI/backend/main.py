import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID")

BASE_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ENGINE_ID}"
)

app = FastAPI()

# --------------------------------------------------
# Auth helper (official Google auth)
# --------------------------------------------------
def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token


# --------------------------------------------------
# Request models
# --------------------------------------------------
class SessionRequest(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


# --------------------------------------------------
# Create session (MANDATORY)
# --------------------------------------------------
@app.post("/api/session")
def create_session(req: SessionRequest):
    token = get_access_token()

    response = requests.post(
        f"{BASE_URL}/sessions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "userId": req.user_id
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()
    session_id = data["name"].split("/")[-1]

    return {"session_id": session_id}


# --------------------------------------------------
# Chat with session
# --------------------------------------------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    token = get_access_token()

    response = requests.post(
        f"{BASE_URL}/sessions/{req.session_id}:query",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "queryInput": {
                "text": {
                    "text": req.message
                },
                "languageCode": "en",
            }
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()

    reply = (
        data.get("queryResult", {})
            .get("responseMessages", [{}])[0]
            .get("text", {})
            .get("text", [""])[0]
    )

    return {"reply": reply}