from fastmcp import FastMCP
import httpx

# -------------------------------------------------
# MCP Server
# -------------------------------------------------

mcp = FastMCP("Weather MCP Server (HTTP-backed)")

BASE_URL = "http://127.0.0.1:8000"

# -------------------------------------------------
# MCP Tools
# -------------------------------------------------

@mcp.tool()
def get_temperature(city: str) -> str:
    """
    Fetch current temperature for a given city from Weather HTTP API
    and return a human-readable text response.
    """
    url = f"{BASE_URL}/temperature/{city}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

        return (
            f"The current temperature in {data['city']} "
            f"is {data['temperature']}{data['unit']}."
        )

    except httpx.HTTPStatusError as e:
        return f"Failed to fetch temperature for {city}. HTTP {e.response.status_code}."

    except Exception as e:
        return f"Error fetching temperature for {city}: {str(e)}"


@mcp.tool()
def get_forecast(city: str) -> str:
    """
    Fetch 5-day weather forecast for a given city from Weather HTTP API
    and return a human-readable text response.
    """
    url = f"{BASE_URL}/forecast/{city}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

        lines = [
            f"5-day weather forecast for {data['city']}:"
        ]

        for day in data["forecast"]:
            lines.append(
                f"- {day['day']}: {day['temperature']}{data['unit']}"
            )

        return "\n".join(lines)

    except httpx.HTTPStatusError as e:
        return f"Failed to fetch forecast for {city}. HTTP {e.response.status_code}."

    except Exception as e:
        return f"Error fetching forecast for {city}: {str(e)}"

# -------------------------------------------------
# Run MCP Server
# -------------------------------------------------

if __name__ == "__main__":
    mcp.run()
