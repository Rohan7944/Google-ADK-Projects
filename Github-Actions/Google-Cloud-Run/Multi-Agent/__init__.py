"""
Google ADK Package Entrypoint.
Exposes the core agent definition to the deployment system wrapper.
"""
from .agent import root_agent

__all__ = ["root_agent"]
