"""
Neo4j knowledge graph store.

Enhancements over baseline:
  - Hierarchical graph model: Document → Chunk → Entity nodes
    (store_document_hierarchy adds Document and Chunk nodes with
     HAS_CHUNK and MENTIONS relationships)
  - Existing flat store_graph() preserved for backward compatibility
"""
from __future__ import annotations

from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jStore:
    """
    Manages Neo4j graph database for knowledge graph construction.

    Supports:
        - Entity node upsert (MERGE semantics)
        - Typed relationship creation
        - Hierarchical document structure (Document → Chunk → Entity)
        - Document provenance tracking
        - Graph queries and traversals
        - Schema constraint setup
    """

    def __init__(
        self,
        uri: str = "bolt://neo4j:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    def connect(self):
        """Establish connection to Neo4j."""
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self._driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {self.uri}")
        self._setup_constraints()

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    # ─── Public API ───────────────────────────────────────────────────────────

    def store_graph(
        self,
        entities: list[dict],
        relationships: list[dict],
        doc_id: str = "",
    ) -> dict:
        """
        Store entities and relationships using MERGE semantics.

        Args:
            entities      : List of entity dicts from RelationshipExtractor.
            relationships : List of relationship dicts.
            doc_id        : Document source identifier for provenance.

        Returns:
            Summary dict with counts of created/merged nodes and edges.
        """
        if self._driver is None:
            self.connect()

        nodes_created = 0
        rels_created = 0

        with self._driver.session(database=self.database) as session:
            for entity in entities:
                result = session.execute_write(self._merge_entity, entity, doc_id)
                nodes_created += result

            for rel in relationships:
                result = session.execute_write(self._merge_relationship, rel, doc_id)
                rels_created += result

        summary = {"nodes_merged": nodes_created, "relationships_merged": rels_created}
        logger.info(f"Graph stored: {summary}")
        return summary

    def store_document_hierarchy(
        self,
        doc_id: str,
        doc_metadata: dict,
        chunks: list[dict],
        entities_by_chunk: dict[str, list[dict]],
    ) -> dict:
        """
        Store a hierarchical Document → Chunk → Entity graph.

        Creates:
          (:Document {id: doc_id, ...metadata...})
          (:Chunk    {id: chunk_id, text: ..., index: ...})
          (:Document)-[:HAS_CHUNK]->(:Chunk)
          (:Chunk)-[:MENTIONS]->(:Entity)

        Args:
            doc_id            : Unique document identifier.
            doc_metadata      : Dict of document-level properties (title, source, …).
            chunks            : List of chunk dicts with keys: id, text, index, char_start, char_end.
            entities_by_chunk : Mapping of chunk_id → list of entity dicts for that chunk.

        Returns:
            Summary dict with node and relationship counts.
        """
        if self._driver is None:
            self.connect()

        doc_nodes = chunk_nodes = entity_nodes = mentions_rels = 0

        with self._driver.session(database=self.database) as session:
            # 1. Create / merge Document node
            doc_nodes += session.execute_write(
                self._merge_document_node, doc_id, doc_metadata
            )

            for chunk in chunks:
                chunk_id = chunk.get("id") or f"{doc_id}_chunk_{chunk.get('index', 0)}"

                # 2. Create / merge Chunk node
                chunk_nodes += session.execute_write(
                    self._merge_chunk_node, chunk_id, chunk, doc_id
                )

                # 3. Create Document → Chunk relationship
                session.execute_write(
                    self._merge_has_chunk, doc_id, chunk_id
                )

                # 4. For each entity in this chunk, merge entity and Chunk → Entity
                for entity in entities_by_chunk.get(chunk_id, []):
                    entity_nodes += session.execute_write(
                        self._merge_entity, entity, doc_id
                    )
                    mentions_rels += session.execute_write(
                        self._merge_mentions, chunk_id, entity.get("id", ""), doc_id
                    )

        summary = {
            "document_nodes": doc_nodes,
            "chunk_nodes": chunk_nodes,
            "entity_nodes": entity_nodes,
            "mentions_relationships": mentions_rels,
        }
        logger.info(f"Hierarchical graph stored for {doc_id!r}: {summary}")
        return summary

    def search_entities(
        self,
        label: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query entities from the graph."""
        if self._driver is None:
            self.connect()

        with self._driver.session(database=self.database) as session:
            if entity_type:
                query = f"MATCH (n:{entity_type}) RETURN n LIMIT $limit"
            else:
                query = "MATCH (n) RETURN n LIMIT $limit"
            result = session.run(query, limit=limit)
            return [dict(record["n"]) for record in result]

    def find_related(self, entity_id: str, depth: int = 2) -> dict:
        """Find all entities related to a given entity within `depth` hops."""
        if self._driver is None:
            self.connect()

        with self._driver.session(database=self.database) as session:
            query = """
            MATCH path = (start {id: $entity_id})-[*1..$depth]-(related)
            RETURN path LIMIT 100
            """
            result = session.run(query, entity_id=entity_id, depth=depth)
            paths = [str(record["path"]) for record in result]
            return {"entity_id": entity_id, "depth": depth, "paths": paths}

    def get_stats(self) -> dict:
        """Return graph statistics (node/edge counts by label)."""
        if self._driver is None:
            self.connect()

        with self._driver.session(database=self.database) as session:
            node_count = session.run(
                "MATCH (n) RETURN count(n) as count"
            ).single()["count"]
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            # Count by label
            label_counts = {}
            for record in session.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt"
            ):
                label_counts[record["label"] or "Unknown"] = record["cnt"]

        return {
            "total_nodes": node_count,
            "total_relationships": rel_count,
            "nodes_by_label": label_counts,
        }

    def clear_graph(self):
        """Remove all nodes and relationships (destructive)."""
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Graph cleared — all nodes and relationships removed")

    # ─── Transaction functions ────────────────────────────────────────────────

    @staticmethod
    def _merge_document_node(tx, doc_id: str, metadata: dict) -> int:
        props = {
            k: str(v) for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool)) and k not in ("pages_text",)
        }
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d += $props
        RETURN d
        """
        result = tx.run(query, doc_id=doc_id, props=props)
        return len(list(result))

    @staticmethod
    def _merge_chunk_node(tx, chunk_id: str, chunk: dict, doc_id: str) -> int:
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.index = $index,
            c.char_start = $char_start,
            c.char_end = $char_end,
            c.doc_id = $doc_id
        RETURN c
        """
        result = tx.run(
            query,
            chunk_id=chunk_id,
            text=chunk.get("text", "")[:2000],  # cap for Neo4j property limits
            index=chunk.get("index", 0),
            char_start=chunk.get("char_start", 0),
            char_end=chunk.get("char_end", 0),
            doc_id=doc_id,
        )
        return len(list(result))

    @staticmethod
    def _merge_has_chunk(tx, doc_id: str, chunk_id: str) -> int:
        query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (c:Chunk    {id: $chunk_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c
        """
        result = tx.run(query, doc_id=doc_id, chunk_id=chunk_id)
        return len(list(result))

    @staticmethod
    def _merge_mentions(tx, chunk_id: str, entity_id: str, doc_id: str) -> int:
        if not entity_id:
            return 0
        query = """
        MATCH (c:Chunk  {id: $chunk_id})
        MATCH (e        {id: $entity_id})
        MERGE (c)-[:MENTIONS {doc_id: $doc_id}]->(e)
        RETURN e
        """
        result = tx.run(query, chunk_id=chunk_id, entity_id=entity_id, doc_id=doc_id)
        return len(list(result))

    @staticmethod
    def _merge_entity(tx, entity: dict, doc_id: str) -> int:
        entity_type = entity.get("type", "Entity")
        entity_id = entity.get("id", "")
        label = entity.get("label", entity_id)
        properties = entity.get("properties", {})

        props_str = ", ".join(
            f"n.{k} = ${k}" for k in properties if k not in ("id", "label")
        )
        set_clause = "SET n.label = $label, n.doc_id = $doc_id"
        if props_str:
            set_clause += f", {props_str}"

        query = f"""
        MERGE (n:{entity_type} {{id: $id}})
        {set_clause}
        RETURN n
        """
        params = {"id": entity_id, "label": label, "doc_id": doc_id, **properties}
        result = tx.run(query, **params)
        return len(list(result))

    @staticmethod
    def _merge_relationship(tx, rel: dict, doc_id: str) -> int:
        source_id = rel.get("source", "")
        target_id = rel.get("target", "")
        rel_type = rel.get("type", "RELATED_TO")
        properties = rel.get("properties", {})

        props_clause = ", ".join(f"{k}: ${k}" for k in properties)
        props_str = (
            f"{{{props_clause}, doc_id: $doc_id}}"
            if properties
            else "{doc_id: $doc_id}"
        )

        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += {props_str}
        RETURN r
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "doc_id": doc_id,
            **properties,
        }
        result = tx.run(query, **params)
        return len(list(result))

    def _setup_constraints(self):
        """Create uniqueness constraints for node IDs."""
        constraints = [
            "CREATE CONSTRAINT entity_id   IF NOT EXISTS FOR (n:Entity)   REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id    IF NOT EXISTS FOR (n:Chunk)    REQUIRE n.id IS UNIQUE",
        ]
        with self._driver.session(database=self.database) as session:
            for cql in constraints:
                try:
                    session.run(cql)
                except Exception:
                    pass  # Constraint already exists
