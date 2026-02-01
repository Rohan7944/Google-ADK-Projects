import asyncio
from fastmcp.client import Client

# -------------------------------------------------
# MCP Client (HTTP)
# -------------------------------------------------

async def main():
    async with Client("http://localhost:8080/mcp") as client:

        # -------------------------------------------------
        # List available tools
        # -------------------------------------------------
        tools = await client.list_tools()

        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        # -------------------------------------------------
        # Call get_temperature
        # -------------------------------------------------
        temperature = await client.call_tool(
            "get_temperature",
            {"city": "Delhi"},
        )

        print("\nTemperature Result:")
        print(temperature)

        # -------------------------------------------------
        # Call get_forecast
        # -------------------------------------------------
        forecast = await client.call_tool(
            "get_forecast",
            {"city": "Mumbai"},
        )

        print("\nForecast Result:")
        print(forecast)


if __name__ == "__main__":
    asyncio.run(main())