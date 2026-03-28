"""Wolfram Alpha Search custom tool for MCP Assist."""
import logging
from typing import Dict, Any, List

import aiohttp

_LOGGER = logging.getLogger(__name__)

WOLFRAM_SHORT_ANSWERS = "https://api.wolframalpha.com/v1/result"
WOLFRAM_FULL_RESULTS  = "https://api.wolframalpha.com/v2/query"


class WolframSearchTool:
    """Wolfram Alpha Search tool for factual and real-world queries."""

    def __init__(self, hass, app_id: str):
        """Initialize Wolfram Alpha Search tool."""
        self.hass = hass
        self.app_id = app_id

    async def initialize(self):
        """Initialize the tool."""
        pass

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this class handles the given tool."""
        return tool_name == "search"

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definition for Wolfram Alpha Search."""
        return [{
            "name": "search",
            "description": (
                "Look up factual, real-world information using Wolfram Alpha. "
                "Use this for: current events, current political office holders, "
                "scientific facts, mathematics, unit conversions, dates, times, "
                "geography, and any question about the state of the world outside "
                "the home. Do NOT use for controlling home devices."
            ),
            "inputSchema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The factual question or query to look up"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }]

    async def handle_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Wolfram Alpha Search."""
        query = arguments.get("query", "").strip()
        if not query:
            return {
                "content": [{
                    "type": "text",
                    "text": "No query provided."
                }]
            }

        _LOGGER.debug(f"Wolfram Alpha query: '{query}'")

        try:
            answer = await self.hass.async_add_executor_job(
                self._query_sync, query
            )

            if answer:
                return {
                    "content": [{
                        "type": "text",
                        "text": answer
                    }]
                }
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"No result found for: {query}"
                    }]
                }

        except Exception as e:
            _LOGGER.error(f"Wolfram Alpha exception: {e}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Wolfram Alpha error: {str(e)}"
                }]
            }

    def _query_sync(self, query: str) -> str:
        """Synchronous Wolfram query — tries Short Answers first, then Full Results."""
        import requests

        # 1. Short Answers endpoint — fast, one plain-text line
        try:
            resp = requests.get(
                WOLFRAM_SHORT_ANSWERS,
                params={"appid": self.app_id, "i": query},
                timeout=8,
            )
            if resp.status_code == 200:
                text = resp.text.strip()
                if text and "did not understand" not in text.lower():
                    return text
        except Exception as exc:
            _LOGGER.warning(f"Wolfram Short Answers failed: {exc}")

        # 2. Full Results endpoint — more thorough fallback
        try:
            resp = requests.get(
                WOLFRAM_FULL_RESULTS,
                params={
                    "appid": self.app_id,
                    "input": query,
                    "format": "plaintext",
                    "output": "JSON",
                },
                timeout=12,
            )
            if resp.status_code != 200:
                return ""

            data = resp.json()
            if not data.get("queryresult", {}).get("success"):
                return ""

            SKIP_PODS = {"Input", "Identity", "InputInterpretation"}
            for pod in data["queryresult"].get("pods", []):
                if pod.get("id") in SKIP_PODS:
                    continue
                for subpod in pod.get("subpods", []):
                    text = subpod.get("plaintext", "").strip()
                    if text:
                        return text
        except Exception as exc:
            _LOGGER.warning(f"Wolfram Full Results failed: {exc}")

        return ""