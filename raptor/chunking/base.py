from abc import ABC, abstractmethod
from typing import List


class Chunker(ABC):
    """Turns a document's raw text into a list of leaf-chunk strings."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        ...
