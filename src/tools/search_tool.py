"""FarmWebSearchTool — real-time web search for prices, laws, and schemes."""

import os
import logging
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("farm-tools")


class SearchInput(BaseModel):
    query: str = Field(
        description="Search query for current farming laws, market prices, mandi rates, or government schemes"
    )


class FarmWebSearchTool:
    """Real-time web search via Tavily API for prices, laws, market info (falls back to DuckDuckGo)."""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    def run(self, query: str) -> str:
        farm_query = f"India farming agriculture {query}"

        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=self.tavily_api_key)
                response = client.search(
                    query=farm_query,
                    max_results=3,
                    include_domains=["agricoop.nic.in", "enam.gov.in", "farmer.gov.in",
                                     "krishijagran.com", "agrifarming.in"],
                )
                results = response.get("results", [])
                if results:
                    output = "Web search results:\n"
                    for r in results[:3]:
                        output += f"\n• {r.get('title', '')}\n  {r.get('content', '')[:300]}...\n"
                    return output
            except Exception as e:
                logger.error(f"Tavily search error: {e}")

        # Fallback: DuckDuckGo (no API key needed)
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            return search.run(farm_query)
        except Exception as e:
            return f"Web search unavailable: {str(e)}. Please check TAVILY_API_KEY."

    def as_tool(self) -> BaseTool:
        search_instance = self

        class _SearchTool(BaseTool):
            name: str = "farm_web_search"
            description: str = (
                "Search the web for current information about: "
                "crop market prices (mandi rates), eNAM platform, legal and illegal crops in India, "
                "where to buy/sell seeds and produce, APMC laws, government scheme updates, "
                "current weather advisories, and any time-sensitive farming information."
            )
            args_schema: Type[BaseModel] = SearchInput

            def _run(self, query: str) -> str:
                return search_instance.run(query)

        return _SearchTool()
