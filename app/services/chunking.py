from __future__ import annotations

import re

from app.domain.models import KnowledgeChunk, KnowledgeDocument


class ParentChildChunker:
    """Split documents for recall while keeping a link to their full parent context."""

    def __init__(self, max_chars: int = 180) -> None:
        self.max_chars = max_chars

    def split(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        sentences = [
            item.strip() for item in re.split(r"(?<=[。；])", document.content) if item.strip()
        ]
        groups: list[str] = []
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > self.max_chars:
                groups.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            groups.append(current)

        metadata = {
            "device_type": document.device_type,
            "fault_type": document.fault_type,
            "alarm_codes": document.alarm_codes,
            "source": document.source,
            "version": document.version,
            "tags": document.tags,
            "parent_content": document.content,
        }
        return [
            KnowledgeChunk(
                id=f"{document.id}-c{index:02d}",
                parent_id=document.id,
                title=document.title,
                content=content,
                metadata=metadata,
            )
            for index, content in enumerate(groups, start=1)
        ]
