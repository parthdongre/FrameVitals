"""
Backward-compatible agent tool imports.

Deprecated: use framevitals.agent_tools instead.
"""

from framevitals.agent_tools import (
    AGENT_TOOLS,
    AgentContext,
    list_tools,
    run_tool,
)

__all__ = [
    "AGENT_TOOLS",
    "AgentContext",
    "list_tools",
    "run_tool",
]
