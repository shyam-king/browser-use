
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
import re
import json

from browser_use.browser import browser
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import os

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList, Controller

llm = ChatOpenAI(model='gpt-4o')
# browser = Browser(config=BrowserConfig(headless=False))



async def test_no_interactive_links():
    browser_context=BrowserContext(
        browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
    )

    task =  """
    go to {url} and extract all links. If there are no links return NO_LINKS, otherwise output links in the following format:
    <links>
    ["link1", "link2", ...]
    <links>
    """.strip()

    url = os.path.abspath("./tests/local_html/no_links.html")
    agent = Agent(
        task=task.format(url=url),
        llm=llm,
        browser_context=browser_context,
        max_actions_per_step=1,
        max_input_tokens=1_000_000
    )
    history: AgentHistoryList = await agent.run(10)

    result = history.final_result()
    assert result is not None
    assert "no_links" in result.lower()

    await browser_context.close()


async def test_interactive_links():
    browser_context=BrowserContext(
        browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
    )
    task = """go to {url} and extract all links to socials like facebook, twitter etc.
    return the links extracted in the following format:
    <links>
    [
        {{ "[social name]": "[link]" }},
        {{ "facebook": "https://..." }},
        ...
    ]
    </links>

    if you found no relevant links, you can put empty array. Ensure proper JSON between <links> tags.
    """.strip()

    url = os.path.abspath("./tests/local_html/links.html")
    agent = Agent(
        task=task.format(url=url),
        llm=llm,
        browser_context=browser_context,
        max_actions_per_step=1,
        max_input_tokens=1_000_000
    )
    history: AgentHistoryList = await agent.run(10)

    result = history.final_result()
    assert result is not None
    assert "<links>" in result.lower()
    assert "google" not in result.lower()


    json_str = re.search(r'<links>(.*?)</links>', result, re.DOTALL).group(1).strip()
    data = json.loads(json_str)
    assert len(data) == 3

    assert "linked.com" in json_str.lower()

    await browser_context.close()

