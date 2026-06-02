import json
import os
import re
from typing import List, Optional

from openai import OpenAI

DEFAULT_MODEL = "openai/gpt-oss-120b:free"

STRUCTURE_PROMPT = """You are an expert document structure parser.
Return ONLY a JSON array of section objects in reading order.
Each object must be: {"title": "<verbatim heading text copied from the document>"}
Rules:
- Titles MUST be copied verbatim from the document so they can be matched exactly.
- Detect sections and subsections; preserve the original wording.
- Do NOT summarize. Do NOT invent headings that are not present in the text.
- If the document has no headings, return []."""


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def parse_titles_from_response(response: Optional[str]) -> List[str]:
    """Extract an ordered list of section titles from a (possibly messy) LLM response."""
    if not response or not response.strip():
        return []
    cleaned = _strip_code_fences(response)
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, list):
        return []
    titles: List[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("title"), str):
            t = item["title"].strip()
            if t:
                titles.append(t)
        elif isinstance(item, str) and item.strip():
            titles.append(item.strip())
    return titles


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def llm_parse_titles(
    text: str, model: str = DEFAULT_MODEL, client: Optional[OpenAI] = None
) -> List[str]:
    """Call the LLM to extract section titles. Network-dependent; mocked in tests."""
    client = client or _make_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": STRUCTURE_PROMPT},
            {"role": "user", "content": text[:120000]},
        ],
    )
    return parse_titles_from_response(completion.choices[0].message.content)
