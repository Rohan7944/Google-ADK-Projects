# -------------------------------------------------
# Config (Cloud Run friendly)
# -------------------------------------------------

import os

USER_DB = {
    "token-basic": {
        "user_id": "user_basic",
        "allowed_tags": {"basic"},
    },
    "token-premium": {
        "user_id": "user_premium",
        "allowed_tags": {"basic", "premium"},
    },
    "token-mid":{
        "user_id": "user_mid",
        "allowed_tags": {},
    },
}

WEATHER_API_BASE_URL = os.environ.get(
    "WEATHER_API_BASE_URL",
    "http://localhost:8000",
)

BEARER_TOKEN = os.environ.get(
    "BEARER_TOKEN",
    "my-secret-token",
)

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

MCP_ACCESS_TOKEN = os.environ.get(
    "MCP_ACCESS_TOKEN",
    "my-token"
)

PORT = int(os.environ.get("PORT", "8080"))