import asyncio
from fastmcp import Client

MCP_SERVER_URL = "http://localhost:8080/mcp"
MCP_ACCESS_TOKEN = "token-mid"  # MUST match server verification

async def main():
    
    try:
        async with Client(
            MCP_SERVER_URL,
            auth=MCP_ACCESS_TOKEN,
        ) as client:
            tools = await client.list_tools()
            print("Available tools:", [t.name for t in tools])

            temp = await client.call_tool("get_temperature", {"city": "delhi"})
            print("Temperature:", temp)

            forecast = await client.call_tool("get_forecast", {"city": "delhi"})
            print("Forecast:", forecast)
    except Exception as e:
        print(f"Errored: {e}")

if __name__ == "__main__":
    asyncio.run(main())
