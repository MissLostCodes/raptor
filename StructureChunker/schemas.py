from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SectionNode:
    title: str
    level: int
    start_page: int
    end_page: Optional[int] = None
    text: str = ""

    children: List["SectionNode"] = field(default_factory=list)


@dataclass
class StructuredChunk:
    chunk_id: str
    title: str
    level: int

    text: str

    start_page: int
    end_page: int

    parent_title: Optional[str] = None