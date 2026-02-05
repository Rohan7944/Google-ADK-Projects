import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.auth
import vertexai
import vertexai.agent_engines as agent_engines

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Environment
# -------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
AGENT_ENGINE_ID = os.getenv("AGENT_ENGINE_ID")

if not all([PROJECT_ID, LOCATION, AGENT_ENGINE_ID]):
    raise RuntimeError("Missing required environment variables")

# Full official resource name
AGENT_ENGINE_RESOURCE = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/{AGENT_ENGINE_ID}"
)

# -------------------------------------------------
# FastAPI
# -------------------------------------------------
app = FastAPI(title="Vertex AI Agent Engine Proxy")

# -------------------------------------------------
# Vertex AI Initialization
# -------------------------------------------------
logger.info("Fetching default Google credentials")
credentials, _ = google.auth.default()

logger.info("Initializing Vertex AI")
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

logger.info("Connecting to Agent Engine: %s", AGENT_ENGINE_RESOURCE)
agent = agent_engines.AgentEngine(
    agent_engine_name=AGENT_ENGINE_RESOURCE
)

# -------------------------------------------------
# Session store (demo only)
# -------------------------------------------------
SESSIONS = {}

# -------------------------------------------------
# Request models
# -------------------------------------------------
class SessionRequest(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.post("/api/session")
def create_session(req: SessionRequest):
    logger.info("Creating agent session for user_id=%s", req.user_id)

    try:
        session = agent.create_session(
            user_id=req.user_id
        )

        SESSIONS[session.session_id] = session

        logger.info("Session created: %s", session.session_id)

        return {
            "session_id": session.session_id
        }

    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat(req: ChatRequest):
    logger.info("Received chat request")

    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    try:
        logger.info("Sending message to agent")
        response = agent.chat(
            session=session,
            message=req.message,
        )

        logger.info("Agent response received")

        return {
            "response": response.text
        }

    except Exception as e:
        logger.exception("Chat call failed")
        raise HTTPException(status_code=500, detail=str(e))