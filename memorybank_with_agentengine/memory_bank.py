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
# from google.adk.tools.load_memory_tool import LoadMemoryTool

from vertexai import types as t
MemoryBankConfig = t.ReasoningEngineContextSpecMemoryBankConfig
CustomizationConfig = t.MemoryBankCustomizationConfig
SimilaritySearchConfig = t.ReasoningEngineContextSpecMemoryBankConfigSimilaritySearchConfig
GenerationConfig = t.ReasoningEngineContextSpecMemoryBankConfigGenerationConfig
TtlConfig = t.ReasoningEngineContextSpecMemoryBankConfigTtlConfig
MemoryTopic = t.MemoryBankCustomizationConfigMemoryTopic
CustomMemoryTopic = t.MemoryBankCustomizationConfigMemoryTopicCustomMemoryTopic
ManagedMemoryTopic = t.MemoryBankCustomizationConfigMemoryTopicManagedMemoryTopic
ManagedTopicEnum = t.ManagedTopicEnum

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

logger.info("Loading environment variables")
load_dotenv()

REQUIRED_ENV_VARS = [
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_GENAI_USE_VERTEXAI", 
]

try:
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required env var: {var}")
except Exception:
    logger.exception("Environment validation failed")
    raise

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") # Valid project needed
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") # Valid location
USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") # Should be set as 1 or True

USER_NAME = "Rohan"
APP_NAME = "simple_chat_agent"
MODEL_NAME = "gemini-2.5-flash"
ENGINE_ID_FILE = "agent_engine_id.txt"

logger.info(
    "Environment validated | project=%s | location=%s | user=%s | USE_VERTEXAI=%s",
    PROJECT_ID,
    LOCATION,
    USER_NAME,
    USE_VERTEXAI,
)

# ------------------------------------------------------------------
# OFFICIAL: Auto-save session to Memory Bank
# ------------------------------------------------------------------

async def auto_save_session_to_memory_callback(callback_context):
    try:
        session_id = callback_context._invocation_context.session.id
        logger.info(
            "Auto-save callback triggered | session_id=%s",
            session_id,
        )

        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
        )

        logger.info(
            "Session successfully saved to Memory Bank | session_id=%s",
            session_id,
        )
    except Exception:
        logger.exception("Auto-save to Memory Bank failed")

# ------------------------------------------------------------------
# Agent Definition (OFFICIAL MEMORY FLOW + RETRIEVAL)
# ------------------------------------------------------------------

logger.info("Creating LLM agent")

agent = LlmAgent(
    name="simple_chat_agent",
    model=MODEL_NAME,
    instruction=(
        "You are a helpful assistant. "
        "When the user shares stable preferences or personal facts, "
        "acknowledge them naturally."
    ),
    tools=[PreloadMemoryTool(),
        #    LoadMemoryTool() # Un-comment if you want to agent to decide whether the memory tool should be invoked.
        ],
    after_agent_callback=auto_save_session_to_memory_callback,
)

logger.info("Agent created successfully")

# ------------------------------------------------------------------
# Agent Engine Handling (ID ONLY)
# ------------------------------------------------------------------

def get_or_create_agent_engine(client):
    try:
        if os.path.exists(ENGINE_ID_FILE):
            logger.info("Agent engine ID file found")
            with open(ENGINE_ID_FILE, "r") as f:
                engine_id = f.read().strip()

            logger.info("Reusing Agent Engine ID: %s", engine_id)
            return engine_id

        logger.info("No existing engine ID found, creating new Agent Engine")
        
        customization_config = CustomizationConfig(
            memory_topics=[
                MemoryTopic(
                    managed_memory_topic=ManagedMemoryTopic(
                        managed_topic_enum=ManagedTopicEnum.USER_PERSONAL_INFO)
                    ),
                MemoryTopic(
                    custom_memory_topic=CustomMemoryTopic(
                        label="business_feedback",
                        description="""Specific user feedback about their experience at
                        the coffee shop. This includes opinions on drinks, food, pastries, ambiance,
                        staff friendliness, service speed, cleanliness, and any suggestions for
                        improvement."""
                    )
                )
            ]
        )
        
        memory_config = MemoryBankConfig(
            customization_configs = [customization_config],
            similarity_search_config = SimilaritySearchConfig(
                embedding_model=(
                    f"projects/{PROJECT_ID}/locations/{LOCATION}"
                    f"/publishers/google/models/text-embedding-005"
                )
            ),
            generation_config=GenerationConfig(
                model=(
                    f"projects/{PROJECT_ID}/locations/{LOCATION}"
                    f"/publishers/google/models/{MODEL_NAME}"
                )
            ),
            ttl_config=TtlConfig(
                default_ttl=f"25920000s" # One month, Granular (per-operation) TTL also available
            )
        )

        engine = client.agent_engines.create( # Can use .update to update the agent engine
            config={
                "display_name": APP_NAME,
                "context_spec": {
                    "memory_bank_config": memory_config
                }
            }
        )

        engine_id = engine.api_resource.name.split("/")[-1]

        try:
            with open(ENGINE_ID_FILE, "w") as f:
                f.write(engine_id)
            logger.info("Agent Engine ID persisted to file")
        except Exception:
            logger.exception("Failed to persist Agent Engine ID")

        logger.info("New Agent Engine created | engine_id=%s", engine_id)
        return engine_id

    except Exception:
        logger.exception("Agent Engine initialization failed")
        raise

# ------------------------------------------------------------------
# Services Initialization
# ------------------------------------------------------------------

def initialize_services():
    try:
        logger.info("Initializing Vertex AI client")
        client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

        agent_engine_id = get_or_create_agent_engine(client)

        logger.info(
            "Initializing session service | engine_id=%s",
            agent_engine_id,
        )
        session_service = VertexAiSessionService(
            project=PROJECT_ID,
            location=LOCATION,
            agent_engine_id=agent_engine_id,
        )

        logger.info(
            "Initializing memory bank service | engine_id=%s",
            agent_engine_id,
        )
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

        logger.info("All services initialized successfully")
        return runner, session_service

    except Exception:
        logger.exception("Service initialization failed")
        raise

# ------------------------------------------------------------------
# Chat Handling (UNCHANGED FLOW)
# ------------------------------------------------------------------

async def generate_response(runner, session_service, user_input):
    logger.info("generate_response called | input_length=%d", len(user_input))

    try:
        logger.info("Creating or reusing session for user=%s", USER_NAME)

        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_NAME,
        )

        logger.info("Session ready | session_id=%s", session.id)

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        logger.info("Invoking agent run loop")
        final_response = None
        event_count = 0

        async for event in runner.run_async(
            user_id=USER_NAME,
            session_id=session.id,
            new_message=user_content,
        ):
            event_count += 1

            if event.is_final_response():
                final_response = event.content.parts[0].text
                logger.info(
                    "Final response received | session_id=%s | events=%d",
                    session.id,
                    event_count,
                )

        if not final_response:
            logger.warning("No final response generated")

        return final_response or "No response generated."

    except Exception:
        logger.exception("Response generation failed")
        return "Sorry, something went wrong."

# ------------------------------------------------------------------
# Streamlit Chat UI
# ------------------------------------------------------------------

logger.info("Initializing Streamlit UI")

st.set_page_config(page_title="Vertex AI Memory Chat Agent", layout="centered")
st.title("💬 Vertex AI Memory Chat Agent")
st.caption(f"User: **{USER_NAME}**")

if "runner" not in st.session_state:
    try:
        logger.info("Bootstrapping services for Streamlit session")
        runner, session_service = initialize_services()
        st.session_state.runner = runner
        st.session_state.session_service = session_service
        st.session_state.chat_history = []
        logger.info("Streamlit session initialized")
    except Exception:
        st.error("Failed to initialize agent.")
        st.stop()

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Say something...")

if user_input:
    logger.info("User submitted message | length=%d", len(user_input))

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = asyncio.run(
                generate_response(
                    st.session_state.runner,
                    st.session_state.session_service,
                    user_input,
                )
            )
            st.markdown(reply)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply}
    )

    logger.info("Assistant response rendered")