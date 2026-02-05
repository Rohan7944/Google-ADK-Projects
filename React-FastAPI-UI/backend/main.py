import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request

# --------------------------------------------------
# Logging configuration (Cloud Run friendly)
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent-engine-proxy")

# --------------------------------------------------
# Load config
# --------------------------------------------------
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID")

logger.info("Starting Agent Engine proxy")
logger.info(f"PROJECT_ID={PROJECT_ID}")
logger.info(f"LOCATION={LOCATION}")
logger.info(f"AGENT_ENGINE_ID={AGENT_ENGINE_ID}")

BASE_URL = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ENGINE_ID}"
)

app = FastAPI()

# --------------------------------------------------
# Auth helper
# --------------------------------------------------
def get_access_token():
    logger.info("Fetching Google access token using ADC")
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    logger.info("Access token fetched successfully")
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
    logger.info(f"Create session request received for user_id={req.user_id}")

    token = get_access_token()

    logger.info("Calling Vertex AI Agent Engine: create session")
    response = requests.post(
        f"{BASE_URL}/sessions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"userId": req.user_id},
        timeout=30,
    )

    logger.info(f"Vertex response status: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"Session creation failed: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create session")

    data = response.json()
    session_id = data["name"].split("/")[-1]

    logger.info(f"Session created successfully: session_id={session_id}")

    return {"session_id": session_id}


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    logger.info(
        f"Chat request received | session_id={req.session_id} | message='{req.message}'"
    )

    token = get_access_token()

    logger.info("Calling Vertex AI Agent Engine: query session")
    response = requests.post(
        f"{BASE_URL}/sessions/{req.session_id}:query",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "queryInput": {
                "text": {"text": req.message},
                "languageCode": "en",
            }
        },
        timeout=60,
    )

    logger.info(f"Vertex response status: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"Chat query failed: {response.text}")
        raise HTTPException(status_code=500, detail="Chat query failed")

    data = response.json()

    reply = (
        data.get("queryResult", {})
        .get("responseMessages", [{}])[0]
        .get("text", {})
        .get("text", [""])[0]
    )

    logger.info(f"Agent reply extracted: '{reply}'")

    return {"reply": reply}