import json
from openai import OpenAI

from .utils import build_page_text
from .schemas import SectionNode
from .config import *

import os

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)


STRUCTURE_PROMPT = """
You are an expert document structure parser.
Extract the hierarchical document structure.
Return ONLY valid JSON.
Format:

[
  {
    "title": "...",
    "level": 1,
    "start_page": 1
  }
]

Rules:
- Detect sections and subsections
- Infer hierarchy
- Preserve original titles
- Detect semantic boundaries
- DO NOT summarize
- DO NOT hallucinate
"""


def extract_document_structure(pages):
    print("Extracting document structure...extract_document_structure(pages) called ")
    document_text = build_page_text(pages)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": STRUCTURE_PROMPT
            },
            {
                "role": "user",
                "content": document_text[:120000]
            }
        ]
    )

    response = completion.choices[0].message.content
    structure = json.loads(response)
    print(f"Structure from llm : {structure}")
    nodes = []

    for item in structure:

        node = SectionNode(
            title=item["title"],
            level=item["level"],
            start_page=item["start_page"]
        )

        nodes.append(node)
    print(f"Nodes : {nodes}")
    return nodes