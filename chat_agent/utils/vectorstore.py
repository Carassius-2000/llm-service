from __future__ import annotations

import json
import struct
from typing import Any

from langchain_community.vectorstores import SQLiteVec
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)


class FixedSQLiteVec(SQLiteVec):
    """Исправленная версия SQLiteVec с корректной обработкой row_factory."""

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        sql_query = f"""
            SELECT 
                text,
                metadata,
                distance
            FROM {self._table} AS e
            INNER JOIN {self._table}_vec AS v on v.rowid = e.rowid  
            WHERE
                v.text_embedding MATCH ?
                AND k = ?
            ORDER BY distance
        """
        cursor = self._connection.cursor()
        cursor.execute(sql_query, [serialize_f32(embedding), k])
        results = cursor.fetchall()

        documents = []
        for row in results:
            if isinstance(row, tuple):
                text, metadata_str, distance = row[0], row[1], row[2]
            else:
                text = row["text"]
                metadata_str = row["metadata"]
                distance = row["distance"]

            metadata = json.loads(metadata_str) if metadata_str else {}
            doc = Document(page_content=text, metadata=metadata)
            documents.append((doc, distance))

        return documents
