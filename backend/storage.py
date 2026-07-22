import json
import sqlite3
from pathlib import Path
from typing import Any
import re
from backend.config import STORAGE_FOLDER


class IndustrialStore:
    def __init__(self, path: Path | None = None):
        self.path = path or STORAGE_FOLDER / "industrialmind.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def clear(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.commit()

    def keyword_search(self, query: str, limit: int = 10):
        query = re.sub(r"[^\w\s]", " ", query)
        query = " ".join(query.split())

        if not query:
            return []
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT chunks.*
                FROM chunks_fts
                JOIN chunks ON chunks.id = chunks_fts.id
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        return [self._row_to_chunk(row) for row in rows]
    

    def add_document(self, document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                insert or replace into documents
                (id, file_name, document_type, source_path, loader, chunks, entities, metadata)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["id"],
                    document["file_name"],
                    document["document_type"],
                    document["source_path"],
                    document.get("loader", ""),
                    document["chunks"],
                    json.dumps(document.get("entities", {})),
                    json.dumps(document.get("metadata", {})),
                ),
            )
            connection.execute(
                "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (document["id"],)
            )
            connection.execute("delete from chunks where document_id = ?", (document["id"],))
          
            connection.executemany(
                """
                insert or replace into chunks
                (id, qdrant_id, document_id, file_name, document_type, text, entities, embedding, position)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["id"],
                        chunk.get("qdrant_id"),
                        chunk["document_id"],
                        chunk["file_name"],
                        chunk["document_type"],
                        chunk["text"],
                        json.dumps(chunk.get("entities", {})),
                        json.dumps(chunk.get("embedding", [])),
                        chunk["position"],
                    )
                    for chunk in chunks
                ],
            )

            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks_fts(id,text,file_name,document_type)
                VALUES(?,?,?,?)
                """,
                [
                    (
                        chunk["id"],
                        chunk["text"],
                        chunk["file_name"],
                        chunk["document_type"],
                    )
                    for chunk in chunks
                ]
            )

    def documents(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("select * from documents order by file_name").fetchall()
        return [
            {
                "id": row["id"],
                "file_name": row["file_name"],
                "document_type": row["document_type"],
                "source_path": row["source_path"],
                "loader": row["loader"],
                "chunks": row["chunks"],
                "entities": json.loads(row["entities"] or "{}"),
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in rows
        ]

    def delete_document(self, document_id: str) -> None:
        """
        Delete a document and all of its chunks.
        """

        with sqlite3.connect(self.path) as connection:

            # Remove FTS records first
            connection.execute(
                """
                DELETE FROM chunks_fts
                WHERE id IN (
                    SELECT id
                    FROM chunks
                    WHERE document_id = ?
                )
                """,
                (document_id,),
            )

            # Remove chunks
            connection.execute(
                """
                DELETE FROM chunks
                WHERE document_id = ?
                """,
                (document_id,),
            )

            # Remove document
            connection.execute(
                """
                DELETE FROM documents
                WHERE id = ?
                """,
                (document_id,),
            )

            connection.commit()

    def chunks(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("select * from chunks order by file_name, position").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def chunks_metadata(self) -> list[dict[str, Any]]:
        """
        Like chunks(), but excludes the embedding column entirely.

        Each embedding is a 768-float vector stored as JSON text; deserializing
        every row's embedding just to count entities or run keyword-based
        agents (metrics, maintenance, compliance, lessons) is pure overhead —
        for a few thousand chunks this can spike memory hard enough to trigger
        an OOM kill (which shows up as an unexplained container crash /
        "segfault" rather than a clean Python error). Use this for anything
        that doesn't need the vector itself.
        """
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                select id, qdrant_id, document_id, file_name, document_type,
                       text, entities, position
                from chunks
                order by file_name, position
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "qdrant_id": row["qdrant_id"],
                "document_id": row["document_id"],
                "file_name": row["file_name"],
                "document_type": row["document_type"],
                "text": row["text"],
                "entities": json.loads(row["entities"] or "{}"),
                "position": row["position"],
            }
            for row in rows
        ]

    def chunks_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"select * from chunks where id in ({placeholders})",
                ids,
            ).fetchall()
        by_id = {row["id"]: self._row_to_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                create table if not exists documents (
                    id text primary key,
                    file_name text not null,
                    document_type text not null,
                    source_path text not null,
                    loader text not null,
                    chunks integer not null,
                    entities text not null,
                    metadata text not null,
                    created_at text default current_timestamp
                )
                """
            )
            connection.execute(
                """
                create table if not exists chunks (
                    id text primary key,
                    qdrant_id integer,
                    document_id text not null,
                    file_name text not null,
                    document_type text not null,
                    text text not null,
                    entities text not null,
                    embedding text not null,
                    position integer not null,
                    foreign key(document_id) references documents(id)
                )
                """
            )
            connection.execute("create index if not exists idx_chunks_document on chunks(document_id)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(file_name)"
            )

            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_position ON chunks(position)"
            )

            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(document_type)"
            )
            connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                id,
                text,
                file_name,
                document_type
            )
            """)

    def _row_to_chunk(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "qdrant_id": row["qdrant_id"],
            "document_id": row["document_id"],
            "file_name": row["file_name"],
            "document_type": row["document_type"],
            "text": row["text"],
            "entities": json.loads(row["entities"] or "{}"),
            "embedding": json.loads(row["embedding"] or "[]"),
            "position": row["position"],
        }


JsonStore = IndustrialStore