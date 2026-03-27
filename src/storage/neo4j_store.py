"""
Neo4j knowledge graph store.
Upgrades: hierarchical Document->Chunk->Entity model via store_document_hierarchy().
"""
from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jStore:
    def __init__(self, uri: str = "bolt://neo4j:7687", username: str = "neo4j",
                 password: str = "password", database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    def connect(self):
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
        if self._driver is None:
            self.connect()
        nodes_created = rels_created = 0
        with self._driver.session(database=self.database) as session:
            for entity in entities:
                nodes_created += session.execute_write(self._merge_entity, entity, doc_id)
            for rel in relationships:
                rels_created += session.execute_write(self._merge_relationship, rel, doc_id)
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
        """Create Document->Chunk->Entity hierarchy with HAS_CHUNK and MENTIONS relationships."""
        if self._driver is None:
            self.connect()
        doc_nodes = chunk_nodes = entity_nodes = mentions = 0
        with self._driver.session(database=self.database) as session:
            doc_nodes += session.execute_write(self._merge_document_node, doc_id, doc_metadata)
            for chunk in chunks:
                chunk_id = chunk.get("id") or f"{doc_id}_chunk_{chunk.get('index', 0)}"
                chunk_nodes += session.execute_write(self._merge_chunk_node, chunk_id, chunk, doc_id)
                session.execute_write(self._merge_has_chunk, doc_id, chunk_id)
                for entity in entities_by_chunk.get(chunk_id, []):
                    entity_nodes += session.execute_write(self._merge_entity, entity, doc_id)
                    mentions += session.execute_write(self._merge_mentions, chunk_id, entity.get("id", ""), doc_id)
        summary = {"document_nodes": doc_nodes, "chunk_nodes": chunk_nodes,
                   "entity_nodes": entity_nodes, "mentions_relationships": mentions}
        logger.info(f"Hierarchical graph stored for {doc_id!r}: {summary}")
        return summary

    def search_entities(self, label: Optional[str] = None, entity_type: Optional[str] = None, limit: int = 50) -> list[dict]:
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.database) as session:
            q = f"MATCH (n:{entity_type}) RETURN n LIMIT $limit" if entity_type else "MATCH (n) RETURN n LIMIT $limit"
            return [dict(r["n"]) for r in session.run(q, limit=limit)]

    def find_related(self, entity_id: str, depth: int = 2) -> dict:
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.database) as session:
            result = session.run(
                "MATCH path=(start {id:$eid})-[*1..$depth]-(r) RETURN path LIMIT 100",
                eid=entity_id, depth=depth,
            )
            return {"entity_id": entity_id, "depth": depth, "paths": [str(r["path"]) for r in result]}

    def get_stats(self) -> dict:
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
            labels = {r["label"] or "Unknown": r["cnt"] for r in session.run(
                "MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt")}
        return {"total_nodes": node_count, "total_relationships": rel_count, "nodes_by_label": labels}

    def clear_graph(self):
        if self._driver is None:
            self.connect()
        with self._driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Graph cleared")

    @staticmethod
    def _merge_document_node(tx, doc_id: str, metadata: dict) -> int:
        props = {k: str(v) for k, v in metadata.items()
                 if isinstance(v, (str, int, float, bool)) and k != "pages_text"}
        result = tx.run("MERGE (d:Document {id:$doc_id}) SET d += $props RETURN d", doc_id=doc_id, props=props)
        return len(list(result))

    @staticmethod
    def _merge_chunk_node(tx, chunk_id: str, chunk: dict, doc_id: str) -> int:
        result = tx.run(
            "MERGE (c:Chunk {id:$cid}) SET c.text=$text, c.index=$idx, c.char_start=$cs, c.char_end=$ce, c.doc_id=$did RETURN c",
            cid=chunk_id, text=chunk.get("text","")[:2000], idx=chunk.get("index",0),
            cs=chunk.get("char_start",0), ce=chunk.get("char_end",0), did=doc_id,
        )
        return len(list(result))

    @staticmethod
    def _merge_has_chunk(tx, doc_id: str, chunk_id: str) -> int:
        result = tx.run(
            "MATCH (d:Document {id:$did}) MATCH (c:Chunk {id:$cid}) MERGE (d)-[:HAS_CHUNK]->(c) RETURN c",
            did=doc_id, cid=chunk_id,
        )
        return len(list(result))

    @staticmethod
    def _merge_mentions(tx, chunk_id: str, entity_id: str, doc_id: str) -> int:
        if not entity_id:
            return 0
        result = tx.run(
            "MATCH (c:Chunk {id:$cid}) MATCH (e {id:$eid}) MERGE (c)-[:MENTIONS {doc_id:$did}]->(e) RETURN e",
            cid=chunk_id, eid=entity_id, did=doc_id,
        )
        return len(list(result))

    @staticmethod
    def _merge_entity(tx, entity: dict, doc_id: str) -> int:
        etype = entity.get("type", "Entity")
        eid   = entity.get("id", "")
        label = entity.get("label", eid)
        props = entity.get("properties", {})
        props_str = ", ".join(f"n.{k}=${k}" for k in props if k not in ("id","label"))
        set_clause = "SET n.label=$label, n.doc_id=$doc_id" + (f", {props_str}" if props_str else "")
        result = tx.run(f"MERGE (n:{etype} {{id:$id}}) {set_clause} RETURN n",
                        id=eid, label=label, doc_id=doc_id, **props)
        return len(list(result))

    @staticmethod
    def _merge_relationship(tx, rel: dict, doc_id: str) -> int:
        src = rel.get("source",""); tgt = rel.get("target","")
        rtype = rel.get("type","RELATED_TO"); props = rel.get("properties",{})
        pc = ", ".join(f"{k}:${k}" for k in props)
        ps = f"{{{pc}, doc_id:$doc_id}}" if props else "{doc_id:$doc_id}"
        result = tx.run(
            f"MATCH (a {{id:$src}}) MATCH (b {{id:$tgt}}) MERGE (a)-[r:{rtype}]->(b) SET r+={ps} RETURN r",
            src=src, tgt=tgt, doc_id=doc_id, **props,
        )
        return len(list(result))

    def _setup_constraints(self):
        stmts = [
            "CREATE CONSTRAINT entity_id   IF NOT EXISTS FOR (n:Entity)   REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id    IF NOT EXISTS FOR (n:Chunk)    REQUIRE n.id IS UNIQUE",
        ]
        with self._driver.session(database=self.database) as session:
            for s in stmts:
                try: session.run(s)
                except Exception: pass
