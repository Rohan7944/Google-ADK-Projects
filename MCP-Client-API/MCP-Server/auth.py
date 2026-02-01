import requests
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import FastMCPError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.exceptions import ToolError

import asyncio

from constants import *

# -------------------------------------------------
# Authentication
# -------------------------------------------------

class UserAuthMiddleware(Middleware):

    async def on_request(self, context: MiddlewareContext, call_next):
        headers = get_http_headers()

        header = headers.get("authorization")
        if not header or not header.startswith("Bearer "):
            raise FastMCPError("Missing or invalid Authorization header")

        token = header.removeprefix("Bearer ").strip()

        user = await self.verify_token(token)
        if not user:
            raise FastMCPError("Invalid bearer token")

        # Attach user info to context (official pattern)
        context.fastmcp_context.set_state("user", user)

        return await call_next(context)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        print("Inside on_list_tools")
        
        user = context.fastmcp_context.get_state("user")
        allowed_tags = user["allowed_tags"]
        tools = await call_next(context)
        print("Tools before filter:", [tool.name for tool in tools])
        if not allowed_tags:
            return tools
        
        filtered_list = []
        for tool in tools:
            tags = getattr(tool,"tags",None) or set()
            print(f"Tool {tool.name} has tags {tags}")
            if set(allowed_tags).intersection(tags):
                filtered_list.append(tool)
        print("Tools after filter:", [tool.name for tool in filtered_list])
        
        return filtered_list

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        print("Inside on_call_tool")
        
        user = context.fastmcp_context.get_state("user")
        allowed_tags = user["allowed_tags"]
        tool_object = await context.fastmcp_context.fastmcp.get_tool(context.message.name)
        tool_tags = tool_object.tags if tool_object else set()
        print(f"User_roles={allowed_tags}, tool={context.message.name}, tool_tags={tool_tags}")
        
        if not set(allowed_tags).intersection(tool_tags):
            raise ToolError(f"Access denied: your roles {allowed_tags} do not match required tags {list(tool_tags)}")

        return await call_next(context)

    async def verify_token(self, token: str):
        await asyncio.sleep(0)
        return USER_DB.get(token)

# -------------------------------------------------
# MCP Server
# -------------------------------------------------

mcp = FastMCP(
    name="weather-mcp-server",
    # description="Weather MCP server for Cloud Run",
)

mcp.add_middleware(UserAuthMiddleware())

# -------------------------------------------------
# Tools
# -------------------------------------------------

@mcp.tool(tags=["basic"])
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


@mcp.tool(tags=["premium"])
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