# [MODULE DOC]
# qwen-agent "search" tool for the Tongyi DeepResearch agent, backed by the Serper
# Google
# Search API. SERPER_KEY is read from env var SERPER_KEY_ID at import (may be None).
import asyncio
import json
import os

import aiohttp
from qwen_agent.tools.base import BaseTool, register_tool

SERPER_KEY = os.environ.get("SERPER_KEY_ID")


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    # [CLASS DOC]
    # qwen-agent BaseTool registered as "search" (overwriting existing registrations).
    # Takes
    # a JSON params object with "query" (string or list of strings) and returns
    # formatted
    # text of top web results per query. All network calls are async via aiohttp.
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: dict | None = None):
        # [FUNCTION DOC]
        # Forwards the optional qwen-agent tool cfg dict to BaseTool.
        super().__init__(cfg)

    async def google_search_with_serp(self, query: str):
        # [FUNCTION DOC]
        # POSTs one query to the Serper search endpoint, using China/zh-cn locale if the
        # query
        # contains CJK ideographs, otherwise United States/en. Up to 5 attempts with a
        # 0.5 s sleep
        # after any exception. On a response: unparsable JSON returns a parse-failure
        # message;
        # missing "organic" returns a no-results hint; otherwise returns a markdown-ish
        # listing
        # "A Google search for ... found N results" with numbered title/link, optional
        # date,
        # source and snippet per result (stripping a video-player boilerplate string).
        # After all
        # attempts fail returns a timeout/error message containing the last exception.
        # Never
        # raises.
        def contains_chinese_basic(text: str) -> bool:
            # [FUNCTION DOC]
            # HELPER: True if any character lies in the CJK Unified Ideographs range
            # U+4E00..U+9FFF.
            return any("\u4e00" <= char <= "\u9fff" for char in text)

        if contains_chinese_basic(query):
            payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        else:
            payload = {"q": query, "location": "United States", "gl": "us", "hl": "en"}

        headers = {"X-API-KEY": SERPER_KEY or "", "Content-Type": "application/json"}

        last_exc = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(5):
                try:
                    async with session.post(
                        "https://google.serper.dev/search",
                        json=payload,
                        headers=headers,
                    ) as resp:
                        text = await resp.text()
                        try:
                            results = json.loads(text)
                        except Exception:
                            return f"[Search] Failed to parse response for '{query}'."

                        if "organic" not in results:
                            return f"No results found for query: '{query}'. Use a less specific query."  # noqa: E501

                        web_snippets = []
                        for idx, page in enumerate(results.get("organic", []), start=1):
                            date_published = (
                                f"\nDate published: {page['date']}"
                                if page.get("date")
                                else ""
                            )
                            source = (
                                f"\nSource: {page['source']}"
                                if page.get("source")
                                else ""
                            )
                            snippet = (
                                f"\n{page['snippet']}" if page.get("snippet") else ""
                            )
                            redacted_version = f"{idx}. [{page.get('title', '')}]({page.get('link', '')}){date_published}{source}\n{snippet}"
                            redacted_version = redacted_version.replace(
                                "Your browser can't play this video.", ""
                            )
                            web_snippets.append(redacted_version)

                        content = (
                            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                            + "\n\n".join(web_snippets)
                        )
                        return content
                except Exception as e:
                    last_exc = e
                    await asyncio.sleep(0.5)
                    continue

        return f"Google search Timeout or error ({last_exc}); return None, Please try again later."  # noqa: E501

    async def search_with_serp(self, query: str):
        # [FUNCTION DOC]
        # HELPER: Thin alias awaiting google_search_with_serp for one query.
        return await self.google_search_with_serp(query)

    async def call(self, params: str | dict, **kwargs) -> str:  # type: ignore[override]
        # [FUNCTION DOC]
        # Tool entry point. Reads params["query"]; on failure (e.g. params is a raw
        # string or
        # lacks the key) returns an invalid-format message. A string query is searched
        # directly;
        # a list is searched concurrently and the results joined by a "======="
        # separator line.
        # Returns str.
        # KNOWN ISSUE: params typed as str | dict but a JSON string is never parsed, and
        # a
        # non-list, non-string query triggers an AssertionError instead of an error
        # message.
        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            return await self.search_with_serp(query)

        assert isinstance(query, list)
        tasks = [self.search_with_serp(q) for q in query]
        responses = await asyncio.gather(*tasks)
        return "\n=======\n".join(responses)
