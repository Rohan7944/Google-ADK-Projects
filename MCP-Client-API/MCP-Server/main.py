import os
import requests
from fastmcp import FastMCP

# -------------------------------------------------
# Config (Cloud Run friendly)
# -------------------------------------------------

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

PORT = int(os.environ.get("PORT", "8080"))

# -------------------------------------------------
# MCP Server
# -------------------------------------------------

mcp = FastMCP(
    name="weather-mcp-server",
    # description="Weather MCP server for Cloud Run",
)

# -------------------------------------------------
# Tools
# -------------------------------------------------

@mcp.tool()
def get_temperature(city: str) -> dict:
    """
    Get current temperature for a city.
    """
    url = f"{WEATHER_API_BASE_URL}/temperature/{city}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {
            "error": str(e),
            "city": city,
        }


@mcp.tool()
def get_forecast(city: str) -> dict:
    """
    Get 5-day weather forecast for a city.
    """
    url = f"{WEATHER_API_BASE_URL}/forecast/{city}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {
            "error": str(e),
            "city": city,
        }

# -------------------------------------------------
# Entry Point (Cloud Run compatible)
# -------------------------------------------------

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=PORT,
        path="/mcp",  # important for MCP clients
    )