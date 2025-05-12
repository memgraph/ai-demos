from gqlalchemy import Memgraph
from storage import Storage
import random
import uuid

from typing import List


class MemgraphStorage(Storage):
    def __init__(self):
        super().__init__()
        self._memgraph = Memgraph()
        self._memgraph.execute("CREATE INDEX ON :All")

    def get_all_categories(self):
        results = self._memgraph.execute_and_fetch(
            """
            MATCH (n) 
            WITH labels(n) AS l
            UNWIND l AS ll
            RETURN DISTINCT ll AS label
            ORDER BY label"""
        )
        return [record["label"] for record in results]

    def get_similar_documents(self, label: str, query_vector: str, n: int):
        results = self._memgraph.execute_and_fetch(
            f"""
            CALL vector_search.search("{label.lower()}_vector_index", {n}, $query_vector)
            YIELD node, similarity
            RETURN node.content AS content, similarity
            """,
            {"query_vector": query_vector},
        )

        return results

    def get_paragraph_ids(self, category: str) -> List[int]:
        ids = list(
            self._memgraph.execute_and_fetch(f"MATCH (p:{category}) RETURN p.id AS id")
        )
        return [x["id"] for x in ids]

    def sample_n_connected_paragraphs(self, category: str, number_of_questions: int):
        ids = self.get_paragraph_ids(category)
        if not len(ids):
            return None

        sample_size = min(number_of_questions, len(ids))

        start_ids = random.sample(ids, k=sample_size)
        results = list(
            self._memgraph.execute_and_fetch(
                f"""
            UNWIND $ids AS id
            MATCH path=(p:{category} {{id: id}})-[:NEXT *bfs 0..5]->(next)
            WITH project(path) as graph
            UNWIND graph.nodes as nodes
            RETURN nodes.content AS content
            ORDER BY nodes.index ASC
            """,
                {"ids": start_ids},
            )
        )
        if len(results) == 0:
            results = list(
                self._memgraph.execute_and_fetch(
                    f"""
                UNWIND $ids AS id
                MATCH (node:{category} {{id: id}})
                RETURN node.content AS content
                ORDER BY node.index ASC
                """,
                    {"ids": start_ids},
                )
            )

        return results

    def ingest_paragraphs(
        self,
        category: str,
        paragraphs: List,
        embeddings: List,
        lang_prefix: str,
        mode: str,
    ):
        if mode == "replace":
            self._memgraph.execute("STORAGE MODE IN_MEMORY_ANALYTICAL")
            self._memgraph.execute("DROP GRAPH")
            self._memgraph.execute("CREATE INDEX ON :All")

        paragraph_nodes = []
        for idx, (text, vector) in enumerate(zip(paragraphs, embeddings)):
            para_id = str(uuid.uuid4())
            vector_list = vector.tolist()
            content = text.strip()

            # Create the paragraph node
            self._memgraph.execute(
                f"""
                CREATE (p:{category}:All {{
                    id: $id,
                    content: $content,
                    page: $page,
                    index: $idx,
                    vector: $vector,
                    lang_prefix: $lang_prefix
                }})
                """,
                {
                    "id": para_id,
                    "content": content,
                    "page": category,
                    "idx": idx,
                    "vector": vector_list,
                    "lang_prefix": lang_prefix,
                },
            )
            paragraph_nodes.append((para_id, idx))

        # Create :NEXT relationships between consecutive paragraphs
        for (id1, _), (id2, _) in zip(paragraph_nodes[:-1], paragraph_nodes[1:]):
            self._memgraph.execute(
                f"""
                MATCH (p1:{category} {{id: $id1}}), (p2:{category} {{id: $id2}})
                CREATE (p1)-[:NEXT]->(p2)
                """,
                {"id1": id1, "id2": id2},
            )

        dimension = len(embeddings[0])
        capacity = len(embeddings) * 2

        index_name = f"{category.lower()}_vector_index"
        self._memgraph.execute(
            f"""
            CREATE VECTOR INDEX {index_name} ON :{category}(vector)
            WITH CONFIG {{
                "dimension": {dimension},
                "capacity": {capacity},
                "metric": "cos"
            }}
        """
        )

        return len(paragraphs)

    def get_all_paragraphs(self, category: str) -> List[str]:
        results = self._memgraph.execute_and_fetch(
            f"""
            MATCH (p:{category})
            RETURN p.content AS content, p.id as id
            ORDER BY p.index ASC
            """
        )
        return [
            {"content": record["content"], "id": record["id"]} for record in results
        ]

    def delete_paragraph(self, category: str, paragraph_id: str):
        self._memgraph.execute(
            """
            MATCH (p {id: $id})
            DETACH DELETE p
            """,
            {"id": paragraph_id},
        )
