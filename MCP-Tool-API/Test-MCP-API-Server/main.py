import requests
from fastmcp import FastMCP

# -------------------------------------------------
# Config
# -------------------------------------------------

WEATHER_API_BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "my-secret-token"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# -------------------------------------------------
# MCP Server
# -------------------------------------------------

mcp = FastMCP(name="weather-mcp-server")


# -------------------------------------------------
# Tools
# -------------------------------------------------

@mcp.tool()
def get_temperature(city: str) -> dict:
    """
    Get current temperature for a city.
    """
    url = f"{WEATHER_API_BASE_URL}/temperature/{city}"

    response = requests.get(url, headers=HEADERS, timeout=10)

    if response.status_code != 200:
        return {
            "error": response.text,
            "status_code": response.status_code,
        }

    return response.json()


@mcp.tool()
def get_forecast(city: str) -> dict:
    """
    Get 5-day weather forecast for a city.
    """
    url = f"{WEATHER_API_BASE_URL}/forecast/{city}"

    response = requests.get(url, headers=HEADERS, timeout=10)

    if response.status_code != 200:
        return {
            "error": response.text,
            "status_code": response.status_code,
        }

    return response.json()


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    mcp.run()