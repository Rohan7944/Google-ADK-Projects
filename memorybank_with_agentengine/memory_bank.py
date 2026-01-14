import os
import asyncio
import logging
import streamlit as st
from dotenv import load_dotenv

import vertexai
from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("simple_chat_agent")

# ------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------

load_dotenv()

REQUIRED_ENV_VARS = [
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required env var: {var}")

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

USER_NAME = "Rohan"
APP_NAME = "simple_chat_agent"
MODEL_NAME = "gemini-2.5-flash"
ENGINE_ID_FILE = "agent_engine_id.txt"

logger.info("Environment validated successfully")

# ------------------------------------------------------------------
# OFFICIAL: Auto-save session to Memory Bank
# ------------------------------------------------------------------

async def auto_save_session_to_memory_callback(callback_context):
    logger.info("Auto-saving session to Memory Bank")
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

# ------------------------------------------------------------------
# Agent Definition (OFFICIAL MEMORY FLOW + RETRIEVAL)
# ------------------------------------------------------------------

agent = LlmAgent(
    name="simple_chat_agent",
    model=MODEL_NAME,
    instruction=(
        "You are a helpful assistant. "
        "When the user shares stable preferences or personal facts, "
        "acknowledge them naturally."
    ),
    tools=[
        PreloadMemoryTool(),  # 🔑 REQUIRED for memory retrieval
    ],
    after_agent_callback=auto_save_session_to_memory_callback,
)

logger.info("Agent created")

# ------------------------------------------------------------------
# Agent Engine Handling (ID ONLY)
# ------------------------------------------------------------------

def get_or_create_agent_engine(client):
    if os.path.exists(ENGINE_ID_FILE):
        with open(ENGINE_ID_FILE, "r") as f:
            engine_id = f.read().strip()
            logger.info(f"Reusing Agent Engine ID: {engine_id}")
            return engine_id

    logger.info("Creating new Agent Engine with Memory Bank enabled")

    engine = client.agent_engines.create(
        config={
            "context_spec": {
                "memory_bank_config": {
                    "generation_config": {
                        "model": (
                            f"projects/{PROJECT_ID}/locations/{LOCATION}"
                            f"/publishers/google/models/{MODEL_NAME}"
                        )
                    }
                }
            }
        }
    )

    engine_id = engine.api_resource.name.split("/")[-1]

    with open(ENGINE_ID_FILE, "w") as f:
        f.write(engine_id)

    logger.info(f"New Agent Engine created: {engine_id}")
    return engine_id

# ------------------------------------------------------------------
# Services Initialization
# ------------------------------------------------------------------

def initialize_services():
    logger.info("Initializing Vertex AI client")
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

    agent_engine_id = get_or_create_agent_engine(client)

    logger.info("Initializing session service")
    session_service = VertexAiSessionService(
        project=PROJECT_ID,
        location=LOCATION,
        agent_engine_id=agent_engine_id,
    )

    logger.info("Initializing memory bank service")
    memory_service = VertexAiMemoryBankService(
        project=PROJECT_ID,
        location=LOCATION,
        agent_engine_id=agent_engine_id,
    )

    logger.info("Initializing runner")
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    logger.info("All services initialized")
    return runner, session_service

# ------------------------------------------------------------------
# Chat Handling (UNCHANGED FLOW)
# ------------------------------------------------------------------

async def generate_response(runner, session_service, user_input):
    try:
        logger.info("Creating or reusing session for user")

        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_NAME,
        )

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        logger.info("Running agent")
        final_response = None

        async for event in runner.run_async(
            user_id=USER_NAME,
            session_id=session.id,
            new_message=user_content,
        ):
            if event.is_final_response():
                final_response = event.content.parts[0].text

        return final_response or "No response generated."

    except Exception:
        logger.exception("Response generation failed")
        return "Sorry, something went wrong."

# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.set_page_config(page_title="Vertex AI Memory Chat Agent", layout="centered")
st.title("💬 Vertex AI Memory Chat Agent")
st.caption(f"User: **{USER_NAME}**")

if "runner" not in st.session_state:
    try:
        runner, session_service = initialize_services()
        st.session_state.runner = runner
        st.session_state.session_service = session_service
    except Exception:
        st.error("Failed to initialize agent.")
        st.stop()

user_input = st.text_input("Say something:")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        reply = asyncio.run(
            generate_response(
                st.session_state.runner,
                st.session_state.session_service,
                user_input,
            )
        )
        st.markdown(f"**Agent:** {reply}")