"""
Neo4j knowledge graph store.

Stores entities and relationships extracted from documents,
with full merge semantics to avoid duplicates across document runs.
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

    def store_graph(self, entities: list[dict], relationships: list[dict], doc_id: str = "") -> dict:
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
            paths = []
            for record in result:
                paths.append(str(record["path"]))
            return {"entity_id": entity_id, "depth": depth, "paths": paths}

    def get_stats(self) -> dict:
        """Return graph statistics (node/edge counts)."""
        if self._driver is None:
            self.connect()

        with self._driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            return {"total_nodes": node_count, "total_relationships": rel_count}

    def clear_graph(self):
        """Remove all nodes and relationships (destructive)."""
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Graph cleared — all nodes and relationships removed")

    # ─── Private transaction functions ───────────────────────────

    @staticmethod
    def _merge_entity(tx, entity: dict, doc_id: str) -> int:
        entity_type = entity.get("type", "Entity")
        entity_id = entity.get("id", "")
        label = entity.get("label", entity_id)
        properties = entity.get("properties", {})

        props_str = ", ".join(
            f"n.{k} = ${k}" for k in properties if k not in ("id", "label")
        )
        set_clause = f"SET n.label = $label, n.doc_id = $doc_id"
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
        props_str = f"{{{props_clause}, doc_id: $doc_id}}" if properties else "{doc_id: $doc_id}"

        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += {props_str}
        RETURN r
        """
        params = {"source_id": source_id, "target_id": target_id, "doc_id": doc_id, **properties}
        result = tx.run(query, **params)
        return len(list(result))

    def _setup_constraints(self):
        """Create uniqueness constraints for entity IDs."""
        with self._driver.session(database=self.database) as session:
            try:
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
            except Exception:
                pass  # Constraint may already exist
