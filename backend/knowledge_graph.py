import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from backend.config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    STORAGE_FOLDER,
)

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


class KnowledgeGraph:

    VALID_RELATIONS = {
            "HAS_COMPONENT",
            "PART_OF",
            "CONNECTED_TO",
            "HAS_FAILURE_MODE",
            "FAILED_DUE_TO",
            "CAUSED_BY",
            "HAS_ROOT_CAUSE",
            "RESOLVED_BY",
            "HAS_CORRECTIVE_ACTION",
            "HAS_PREVENTIVE_ACTION",
            "INSPECTED_BY",
            "HAS_FINDING",
            "REQUIRES_MAINTENANCE",
            "UNDERWENT_MAINTENANCE",
            "HAS_PARAMETER",
            "OPERATES_AT",
            "LOCATED_AT",
            "MONITORED_BY",
            "REFERENCES",
            "GOVERNED_BY",
            "COMPLIES_WITH",
            "HAS_HAZARD",
            "USES",
            "MANUFACTURED_BY",
            "REPLACED_WITH",
            "SIMILAR_TO",
        }

    def __init__(self, path: Path | None = None):

        self.path = (
            path
            or STORAGE_FOLDER / "knowledge_graph.json"
        )

        self.driver = None

        self._constraints_ready = False

        if GraphDatabase:

            try:

                self.driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(
                        NEO4J_USERNAME,
                        NEO4J_PASSWORD,
                    ),
                )

            except Exception as e:

                print(
                    "[KnowledgeGraph] "
                    f"Neo4j driver init failed: {e}"
                )

                self.driver = None

    # ========================================================
    # BUILD FULL NETWORKX / JSON GRAPH
    # ========================================================

    def build(
        self,
        documents: list[dict[str, Any]]
    ) -> dict[str, Any]:

        """
        Build NetworkX/JSON graph from all stored documents.

        Used by /knowledge-graph and after clearing
        the knowledge base.
        """

        graph = nx.MultiDiGraph()

        for document in documents:

            self._add_document_to_graph(
                graph,
                document
            )

        payload = self._graph_to_payload(
            graph,
            len(documents)
        )

        self.path.write_text(
            json.dumps(
                payload,
                indent=2
            ),
            encoding="utf-8",
        )

        return payload

    # ========================================================
    # ADD ONE NEW DOCUMENT
    # ========================================================

    def add_document(
        self,
        document: dict[str, Any],
        all_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:

        """
        Rebuild JSON graph for UI from all documents,
        but write only the newly uploaded document
        to Neo4j.
        """

        graph = nx.MultiDiGraph()

        for doc in all_documents:

            self._add_document_to_graph(
                graph,
                doc
            )

        payload = self._graph_to_payload(
            graph,
            len(all_documents)
        )

        self.path.write_text(
            json.dumps(
                payload,
                indent=2
            ),
            encoding="utf-8",
        )

        # Incremental Neo4j write
        self._write_neo4j_single(
            document
        )

        return payload

    # ========================================================
    # ADD DOCUMENT TO NETWORKX GRAPH
    # ========================================================

    def _add_document_to_graph(
        self,
        graph: nx.MultiDiGraph,
        document: dict[str, Any],
    ) -> None:

        # ----------------------------------------------------
        # Document node
        # ----------------------------------------------------

        doc_node = (
            f"doc:{document['id']}"
        )

        graph.add_node(
            doc_node,
            label=document["file_name"],
            type="Document",
        )

        # ----------------------------------------------------
        # Context-aware entities extracted by LLM
        # ----------------------------------------------------

        knowledge_entities = document.get(
            "knowledge_entities",
            []
        )

        for entity in knowledge_entities:

            name = str(
                entity.get(
                    "name",
                    ""
                )
            ).strip()

            entity_type = str(
                entity.get(
                    "type",
                    "Entity"
                )
            ).strip()

            if not name:
                continue

            if entity_type.lower() in {
                "date",
                "dates",
                "page",
                "number"
            }:
                continue

            if name.lower() in {
                "maintenance",
                "inspection",
                "equipment",
                "failure",
                "procedure",
                "system",
                "document"
            }:
                continue

            equipment_types = {
                    "equipment",
                    "equipment_tags",
                    "asset",
                    "assets"
                }

            if entity_type.lower() in equipment_types:
                entity_type = "Asset"

            node_id = self._entity_id(
                entity_type,
                name
            )
            

            # graph.add_node(
            #     node_id,
            #     label=name,
            #     type=entity_type,
            # )

            # # Document mentions entity
            # graph.add_edge(
            #     doc_node,
            #     node_id,
            #     relation="MENTIONS",
            #     edge_type="DOCUMENT_REFERENCE"
            # )

            if entity_type == "Asset":

                if node_id in graph.nodes:

                    graph.nodes[node_id]["documents"] += 1

                else:

                    graph.add_node(
                        node_id,
                        label=name,
                        type="Asset",
                        equipment_tag=name,
                        documents=1
                    )

                graph.add_edge(
                    doc_node,
                    node_id,
                    relation="HAS_ASSET",
                    edge_type="ASSET_REFERENCE"
                )

            else:

                graph.add_node(
                    node_id,
                    label=name,
                    type=entity_type,
                )

                graph.add_edge(
                    doc_node,
                    node_id,
                    relation="MENTIONS",
                    edge_type="DOCUMENT_REFERENCE"
                )

        # ----------------------------------------------------
        # Fallback regex entities
        #
        # This ensures entities such as equipment tags,
        # dates and process parameters are not lost if
        # the LLM did not extract them.
        # ----------------------------------------------------

        for entity_type, values in document.get(
            "entities",
            {}
        ).items():

            for value in values:

                value = str(value).strip()

                if not value:
                    continue

                if entity_type.lower() in {
                    "dates",
                    "failure_modes",
                    "process_parameters"
                }:
                    continue

                if value.lower() in {
                    "maintenance",
                    "inspection",
                    "equipment",
                    "failure",
                    "procedure",
                    "system",
                    "document"
                }:
                    continue

                # Check whether LLM already created this entity
                existing_node = self._find_entity_node(
                    graph,
                    value
                )

                if existing_node:

                    graph.add_edge(
                        doc_node,
                        existing_node,
                        relation="MENTIONS",
                        edge_type="DOCUMENT_REFERENCE"
                    )

                    continue

                equipment_types = {
                    "equipment",
                    "equipment_tags",
                    "asset",
                    "assets"
                }

                if entity_type.lower() in equipment_types:
                    entity_type = "Asset"

                node_id = self._entity_id(
                    entity_type,
                    value
                )

                # graph.add_node(
                #     node_id,
                #     label=value,
                #     type=entity_type,
                # )

                # graph.add_edge(
                #     doc_node,
                #     node_id,
                #     relation="MENTIONS",
                #     edge_type="DOCUMENT_REFERENCE"
                # )

                if entity_type == "Asset":

                    if node_id in graph.nodes:

                        graph.nodes[node_id]["documents"] += 1

                    else:

                        graph.add_node(
                            node_id,
                            label=value,
                            type="Asset",
                            equipment_tag=value,
                            documents=1
                        )

                    graph.add_edge(
                        doc_node,
                        node_id,
                        relation="HAS_ASSET",
                        edge_type="ASSET_REFERENCE"
                    )

                else:

                    graph.add_node(
                        node_id,
                        label=value,
                        type=entity_type,
                    )

                    graph.add_edge(
                        doc_node,
                        node_id,
                        relation="MENTIONS",
                        edge_type="DOCUMENT_REFERENCE"
                    )

        # ----------------------------------------------------
        # Actual semantic relationships extracted by LLM
        # ----------------------------------------------------

        for relationship in document.get(
            "relationships",
            []
        ):

            source_name = str(
                relationship.get(
                    "source",
                    ""
                )
            ).strip()

            target_name = str(
                relationship.get(
                    "target",
                    ""
                )
            ).strip()

            relation = str(
                relationship.get(
                    "type",
                    "RELATED_TO"
                )
            ).strip().upper()

            if relation not in self.VALID_RELATIONS:
                continue

            if (
                not source_name
                or not target_name
            ):
                continue

            source_id = self._find_entity_node(
                graph,
                source_name
            )

            target_id = self._find_entity_node(
                graph,
                target_name
            )

            # Only create relationship when
            # both entities actually exist
            if source_id and target_id:

                graph.add_edge(
                    source_id,
                    target_id,
                    relation=relation,
                    document=document["file_name"],
                    document_id=document["id"],
                    source_type=graph.nodes[source_id]["type"],
                    target_type=graph.nodes[target_id]["type"],
                )

    # ========================================================
    # FIND ENTITY NODE
    # ========================================================

    def _find_entity_node(
        self,
        graph: nx.MultiDiGraph,
        name: str,
    ) -> str | None:

        normalized_name = (
            name
            .lower()
            .strip()
        )

        for node, attrs in graph.nodes(
            data=True
        ):

            # Do not accidentally match document names
            if attrs.get("type") == "Document":
                continue

            label = str(
                attrs.get(
                    "label",
                    ""
                )
            ).lower().strip()

            if label == normalized_name:

                return node

        return None
    
    def _normalize_asset_name(self, name: str) -> str:
        """
        Normalize equipment names so duplicates become one node.
        """

        name = name.upper().strip()

        name = re.sub(r"\s+", "", name)

        name = name.replace("-", "")

        return name

    # ========================================================
    # CREATE STABLE ENTITY ID
    # ========================================================

    def _entity_id(
        self,
        entity_type: str,
        name: str,
    ) -> str:
        
        if entity_type == "Asset":
            name = self._normalize_asset_name(name)

        return (
            f"{entity_type}:"
            f"{name}"
        )

    # ========================================================
    # NETWORKX -> JSON
    # ========================================================

    def _graph_to_payload(
        self,
        graph: nx.MultiDiGraph,
        document_count: int,
    ) -> dict[str, Any]:

        return {

            "nodes": [

                {
                    "id": node,
                    **attrs
                }

                for node, attrs
                in graph.nodes(
                    data=True
                )
            ],

            "edges": [

                {
                    "source": source,
                    "target": target,
                    "type": attrs.get("relation", ""),
                    **attrs
                }

                for (
                    source,
                    target,
                    _key,
                    attrs
                )

                in graph.edges(
                    keys=True,
                    data=True
                )
            ],

            "stats": {

                "nodes":
                    graph.number_of_nodes(),

                "edges":
                    graph.number_of_edges(),

                "documents":
                    document_count,
            },
        }

    # ========================================================
    # NEO4J CONSTRAINTS
    # ========================================================

    def _ensure_constraints(
        self,
        session
    ) -> None:

        if self._constraints_ready:
            return

        session.run(
            """
            CREATE CONSTRAINT document_id
            IF NOT EXISTS
            FOR (d:Document)
            REQUIRE d.id IS UNIQUE
            """
        )

        session.run(
            """
            CREATE CONSTRAINT entity_id
            IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.id IS UNIQUE
            """
        )

        self._constraints_ready = True

    # ========================================================
    # WRITE ONE DOCUMENT TO NEO4J
    # ========================================================

    def _write_neo4j_single(
        self,
        document: dict[str, Any]
    ) -> None:

        if not self.driver:
            return

        try:

            with self.driver.session() as session:

                self._ensure_constraints(
                    session
                )

                # --------------------------------------------
                # Create document
                # --------------------------------------------

                metadata = document.get("metadata", {}) or {}
                if isinstance(metadata, dict):
                    metadata_json = json.dumps(metadata)
                else:
                    metadata_json = json.dumps({})

                session.run(
                    """
                    MERGE (d:Document {id: $id})

                    SET
                        d.file_name = $file_name,
                        d.document_type = $document_type,
                        d.source_path = $source_path,
                        d.metadata_json = $metadata_json
                    """,
                    metadata_json=metadata_json,
                    id=document["id"],

                    file_name=document[
                        "file_name"
                    ],

                    document_type=document[
                        "document_type"
                    ],

                    source_path=document.get(
                        "source_path",
                        ""
                    ),
                )

                # --------------------------------------------
                # Create contextual LLM entities
                # --------------------------------------------

                for entity in document.get(
                    "knowledge_entities",
                    []
                ):

                    name = str(
                        entity.get(
                            "name",
                            ""
                        )
                    ).strip()

                    entity_type = str(
                        entity.get(
                            "type",
                            "Entity"
                        )
                    ).strip()

                    if not name:
                        continue

                    entity_id = self._entity_id(
                        entity_type,
                        name
                    )

                    session.run(
                        """
                        MERGE (
                            e:Entity {
                                id: $entity_id
                            }
                        )

                        SET
                            e.name = $name,
                            e.entity_type = $entity_type

                        WITH e

                        MATCH (
                            d:Document {
                                id: $document_id
                            }
                        )

                        MERGE (
                            d
                        )-[:MENTIONS]->(
                            e
                        )
                        """,

                        entity_id=entity_id,

                        name=name,

                        entity_type=entity_type,

                        document_id=document[
                            "id"
                        ],
                    )

                # --------------------------------------------
                # Create fallback regex entities
                # --------------------------------------------

                for entity_type, values in document.get(
                    "entities",
                    {}
                ).items():

                    for value in values:

                        value = str(
                            value
                        ).strip()

                        if not value:
                            continue

                        entity_id = self._entity_id(
                            entity_type,
                            value
                        )

                        session.run(
                            """
                            MERGE (
                                e:Entity {
                                    id: $entity_id
                                }
                            )

                            SET
                                e.name = $name,
                                e.entity_type = $entity_type

                            WITH e

                            MATCH (
                                d:Document {
                                    id: $document_id
                                }
                            )

                            MERGE (
                                d
                            )-[:MENTIONS]->(
                                e
                            )
                            """,

                            entity_id=entity_id,

                            name=value,

                            entity_type=entity_type,

                            document_id=document[
                                "id"
                            ],
                        )

                # --------------------------------------------
                # Create semantic relationships
                # --------------------------------------------

                for relationship in document.get(
                    "relationships",
                    []
                ):

                    source_name = str(
                        relationship.get(
                            "source",
                            ""
                        )
                    ).strip()

                    target_name = str(
                        relationship.get(
                            "target",
                            ""
                        )
                    ).strip()

                    relation = str(
                        relationship.get(
                            "type",
                            "RELATED_TO"
                        )
                    ).strip().upper()

                    if relation not in self.VALID_RELATIONS:
                        continue

                    if (
                        not source_name
                        or not target_name
                    ):
                        continue

                    # Find entities by name.
                    #
                    # This assumes LLM relationship names
                    # match extracted entity names.

                    # Neo4j relationship types cannot be
                    # passed as normal query parameters.
                    # Sanitize before inserting into query.

                    safe_relation = (
                        self._safe_relationship_type(
                            relation
                        )
                    )

                    query = f"""
                    MATCH (
                        source:Entity
                    )

                    WHERE
                        toLower(source.name)
                        =
                        toLower($source_name)

                    MATCH (
                        target:Entity
                    )

                    WHERE
                        toLower(target.name)
                        =
                        toLower($target_name)

                    MERGE (
                        source
                    )-[:{safe_relation}]->(
                        target
                    )
                    """

                    session.run(
                        query,

                        source_name=source_name,

                        target_name=target_name,
                    )

        except Exception as e:

            print(
                "[KnowledgeGraph] "
                "Neo4j write failed "
                f"(will retry next call): {e}"
            )

    # ========================================================
    # SAFE NEO4J RELATIONSHIP TYPE
    # ========================================================

    def _safe_relationship_type(
        self,
        relation: str
    ) -> str:

        relation = relation.upper()

        relation = re.sub(
            r"[^A-Z0-9_]",
            "_",
            relation
        )

        relation = relation.strip(
            "_"
        )

        if not relation:

            return "RELATED_TO"

        # Neo4j relationship type should not
        # begin with a number
        if relation[0].isdigit():

            relation = (
                "REL_" + relation
            )

        return relation

    # ========================================================
    # FULL NEO4J SYNC
    # ========================================================

    def _write_neo4j(
        self,
        documents: list[dict[str, Any]]
    ) -> None:

        """
        Full Neo4j synchronization.

        Use only when a complete re-sync
        is explicitly required.
        """

        if not self.driver:
            return

        for document in documents:

            self._write_neo4j_single(
                document
            )