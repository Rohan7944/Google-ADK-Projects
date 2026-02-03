import asyncio
from fastmcp import Client

MCP_SERVER_URL = "http://localhost:8080/mcp"
MCP_ACCESS_TOKEN = "token-premium"  # Change your access token here

async def main():
    
    try:
        async with Client(
            MCP_SERVER_URL,
            auth=MCP_ACCESS_TOKEN,
        ) as client:
            
            try:
                tools = await client.list_tools()
                print("Available tools:", [t.name for t in tools])
            except Exception as e:
                print(f"Errored while accessing list tools: {e}")
                
            try:
                temp = await client.call_tool("get_temperature", {"city": "delhi"})
                print("Temperature:", temp)
            except Exception as e:
                print(f"Errored while accessing get_temperature tool: {e}")

            try:
                forecast = await client.call_tool("get_forecast", {"city": "delhi"})
                print("Forecast:", forecast)
            except Exception as e:
                print(f"Errored while accessing get_forecast tool: {e}")
                
    except Exception as e:
        print(f"Errored while connecting to MCP server: {e}")

if __name__ == "__main__":
    asyncio.run(main())
