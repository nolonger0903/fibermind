from __future__ import annotations

import json
from pathlib import Path

from app.domain.models import KnowledgeDocument


class JsonKnowledgeRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_documents(self) -> list[KnowledgeDocument]:
        raw_documents = json.loads(self.path.read_text(encoding="utf-8"))
        return [KnowledgeDocument(**item) for item in raw_documents]


class StandardAnswerRepository:
    def __init__(self, path: Path) -> None:
        self.answers: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))

    def match(self, query: str, intent: str) -> dict[str, object] | None:
        from app.utils.text import normalize_text

        normalized_query = normalize_text(query)
        for item in self.answers:
            if item["intent"] != intent:
                continue
            for pattern in item["patterns"]:
                normalized_pattern = normalize_text(str(pattern))
                if normalized_pattern in normalized_query or normalized_query in normalized_pattern:
                    return item
        return None
