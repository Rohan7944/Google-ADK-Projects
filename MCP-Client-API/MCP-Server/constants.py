# -------------------------------------------------
# Config (Cloud Run friendly)
# -------------------------------------------------

import os

WEATHER_API_BASE_URL = os.environ.get(
    "WEATHER_API_BASE_URL",
    "http://localhost:9000",
)

BEARER_TOKEN = os.environ.get(
    "BEARER_TOKEN",
    "my-secret-token",
)

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

PORT = int(os.environ.get("PORT", "8080"))