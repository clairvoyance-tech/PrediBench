import json
import os
import urllib.parse
from datetime import date
from typing import Literal, Tuple

import requests
from markdownify import markdownify as md
from predibench.logger_config import get_logger

logger = get_logger(__name__)


def web_search_common(
    query: str,
    provider: Literal["serpapi", "bright_data", "serper"],
    cutoff_date: date | None,
) -> Tuple[str, list[str]]:
    """Perform a web search via the selected provider and return a markdown summary and sources.

    Returns (markdown, sources).
    """
    if provider == "serpapi":
        organic_key = "organic_results"
        api_key = os.getenv("SERPAPI_API_KEY")
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "google_domain": "google.com",
        }
        if cutoff_date is not None:
            params["tbs"] = f"cdr:1,cd_max:{cutoff_date.strftime('%m/%d/%Y')}"
        response = requests.get("https://serpapi.com/search.json", params=params)
    elif provider == "bright_data":
        organic_key = "organic"
        api_key = os.getenv("BRIGHT_SERPER_API_KEY")

        search_params = {
            "q": query,
            "gl": "US",
            "hl": "en",
            "brd_json": "1",
        }
        if cutoff_date is not None:
            search_params["tbs"] = f"cdr:1,cd_max:{cutoff_date.strftime('%m/%d/%Y')}"

        encoded_params = urllib.parse.urlencode(search_params)
        search_url = f"https://www.google.com/search?{encoded_params}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "method": "GET",
            "zone": "serp_api1",
            "url": search_url,
            "format": "json",
        }
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
    elif provider == "serper":
        organic_key = "organic"
        api_key = os.getenv("SERPER_API_KEY")
        payload = {"q": query}
        if cutoff_date is not None:
            payload["tbs"] = f"cdr:1,cd_max:{cutoff_date.strftime('%m/%d/%Y')}"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    if response.status_code != 200:
        logger.error(f"Error response: {response.status_code}")
        logger.error(f"Response text: {response.text}")
        raise ValueError(response.text)

    if provider == "bright_data":
        raw_result = response.json()
        results = json.loads(raw_result.get("body", "{}"))
    else:
        results = response.json()

    if organic_key not in results:
        raise ValueError(f"No results found for query: '{query}'. Use a less restrictive query.")

    if not results.get(organic_key):
        return f"No results found for '{query}'. Try with a more general query.", []

    web_snippets: list[str] = []
    sources: list[str] = []

    for idx, page in enumerate(results[organic_key]):
        date_published = ""
        if provider == "bright_data" and isinstance(page.get("extensions"), list) and page["extensions"]:
            first_ext = page["extensions"][0]
            if isinstance(first_ext, dict) and "text" in first_ext:
                date_published = "\nDate published: " + first_ext["text"]
        elif "date" in page:
            date_published = "\nDate published: " + page["date"]

        source = "\nSource: " + page["source"] if "source" in page else ""

        if provider == "bright_data" and "description" in page:
            snippet = "\n" + page["description"]
        else:
            snippet = "\n" + page.get("snippet", "")

        link = page.get("link", "")
        title = page.get("title", link)
        redacted = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
        web_snippets.append(redacted)
        if link:
            sources.append(link)

    sources = list(dict.fromkeys(sources))
    return f"## Search Results for '{query}'\n" + "\n\n".join(web_snippets), sources


def visit_webpage_scrapfly(url: str, asp: bool = True, render_js: bool = True) -> Tuple[str, list[str]]:
    """Fetch a webpage via Scrapfly and return markdown content and sources list.

    Returns (markdown, [url]).
    """
    api_key = os.getenv("SCRAPFLY_API_KEY")
    if not api_key:
        raise ValueError("Missing SCRAPFLY_API_KEY environment variable for Scrapfly API.")

    from scrapfly import ScrapflyClient, ScrapeConfig, ScrapeApiResponse  # lazy import

    scrapfly = ScrapflyClient(key=api_key)
    result: ScrapeApiResponse = scrapfly.scrape(
        ScrapeConfig(tags=["player", "project:default"], asp=asp, render_js=render_js, url=url)
    )
    html_content = result.content
    markdown_content = md(html_content, heading_style="ATX")
    return markdown_content

