"""SIC Industry Group embedding generator."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        "Please install sentence-transformers: "
        "pip install sentence-transformers"
    )
    raise

try:
    from gqlalchemy import Memgraph
except ImportError:
    Memgraph = None
    print(
        "Warning: gqlalchemy not installed. "
        "Database operations unavailable."
    )


DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class IndustryGroupEmbedding:
    """Represents an IndustryGroup node together with its embedding context."""

    ig_code: str
    ig_name: str
    mg_code: str
    mg_name: str
    mg_description: str
    div_code: str
    div_name: str
    context_text: str
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ig_code": self.ig_code,
            "ig_name": self.ig_name,
            "mg_code": self.mg_code,
            "mg_name": self.mg_name,
            "div_code": self.div_code,
            "div_name": self.div_name,
            "context_text": self.context_text,
            "embedding": self.embedding,
        }


def build_context_text(
    div_name: str,
    mg_name: str,
    mg_description: str,
    ig_name: str,
) -> str:
    """Build a rich context string for embedding generation."""
    parts = [
        f"Division: {div_name}",
        f"Major Group: {mg_name}",
        f"Industry Group: {ig_name}",
    ]

    if mg_description:
        description = mg_description.replace("SIC Search ", "").strip()
        if len(description) > 500:
            description = description[:500] + "..."
        parts.append(f"Description: {description}")

    return " | ".join(parts)


def load_industry_groups_from_json(
    json_path: str,
) -> list[IndustryGroupEmbedding]:
    """Load IndustryGroup records from the JSON export file."""
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    industry_groups = []
    for division in data:
        for major_group in division.get("major_groups", []):
            for industry_group in major_group.get("industry_groups", []):
                context_text = build_context_text(
                    division["name"],
                    major_group["name"],
                    major_group.get("description", ""),
                    industry_group["name"],
                )
                industry_groups.append(
                    IndustryGroupEmbedding(
                        ig_code=industry_group["code"],
                        ig_name=industry_group["name"],
                        mg_code=major_group["code"],
                        mg_name=major_group["name"],
                        mg_description=major_group.get("description", ""),
                        div_code=division["code"],
                        div_name=division["name"],
                        context_text=context_text,
                    )
                )

    return industry_groups


def load_industry_groups_from_memgraph(
    host: str,
    port: int,
) -> list[IndustryGroupEmbedding]:
    """Load IndustryGroup records from Memgraph."""
    if Memgraph is None:
        raise ImportError("gqlalchemy is required for database operations")

    db = Memgraph(host=host, port=port)
    query = """
    MATCH (d:Division)-[:HAS_MAJOR_GROUP]->(mg:MajorGroup)
          -[:HAS_INDUSTRY_GROUP]->(ig:IndustryGroup)
    RETURN d.code AS div_code, d.name AS div_name,
           mg.code AS mg_code, mg.name AS mg_name,
           mg.description AS mg_description,
           ig.code AS ig_code, ig.name AS ig_name
    ORDER BY ig.code
    """

    industry_groups = []
    for row in db.execute_and_fetch(query):
        context_text = build_context_text(
            row["div_name"],
            row["mg_name"],
            row["mg_description"] or "",
            row["ig_name"],
        )
        industry_groups.append(
            IndustryGroupEmbedding(
                ig_code=row["ig_code"],
                ig_name=row["ig_name"],
                mg_code=row["mg_code"],
                mg_name=row["mg_name"],
                mg_description=row["mg_description"] or "",
                div_code=row["div_code"],
                div_name=row["div_name"],
                context_text=context_text,
            )
        )

    return industry_groups


def generate_embeddings(
    industry_groups: list[IndustryGroupEmbedding],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[IndustryGroupEmbedding]:
    """Generate embeddings for all IndustryGroup contexts."""
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [industry_group.context_text for industry_group in industry_groups]
    print(f"Generating embeddings for {len(texts)} industry groups...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    for industry_group, embedding in zip(industry_groups, embeddings):
        industry_group.embedding = embedding.tolist()

    print(
        f"Generated {len(embeddings)} embeddings of dimension "
        f"{len(embeddings[0])}"
    )
    return industry_groups


def save_embeddings_to_memgraph(
    industry_groups: list[IndustryGroupEmbedding],
    host: str,
    port: int,
) -> None:
    """Save embeddings directly to Memgraph."""
    if Memgraph is None:
        raise ImportError("gqlalchemy is required for database operations")

    db = Memgraph(host=host, port=port)
    print(f"Saving embeddings to Memgraph at {host}:{port}...")

    for industry_group in industry_groups:
        query = """
        MATCH (ig:IndustryGroup {code: $code})
        SET ig.embedding = $embedding,
            ig.context_text = $context_text
        """
        db.execute(
            query,
            {
                "code": industry_group.ig_code,
                "embedding": industry_group.embedding,
                "context_text": industry_group.context_text,
            },
        )

    print(f"Saved embeddings for {len(industry_groups)} industry groups")


def save_embeddings_to_json(
    industry_groups: list[IndustryGroupEmbedding],
    output_path: str,
) -> None:
    """Save embeddings to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            [industry_group.to_dict() for industry_group in industry_groups],
            handle,
            indent=2,
        )
    print(f"Saved embeddings to {output_path}")


def save_embeddings_to_cypherl(
    industry_groups: list[IndustryGroupEmbedding],
    output_path: str,
) -> None:
    """Save embeddings as Cypher statements for loading into Memgraph."""
    queries = []
    for industry_group in industry_groups:
        context_text = industry_group.context_text.replace("'", "\\'")
        embedding_json = json.dumps(industry_group.embedding)
        queries.append(
            f"MATCH (ig:IndustryGroup {{code: '{industry_group.ig_code}'}}) "
            f"SET ig.embedding = {embedding_json}, "
            f"ig.context_text = '{context_text}';"
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(queries))
    print(f"Saved Cypher queries to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for SIC IndustryGroup nodes"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--from-json",
        type=str,
        help="Load IndustryGroup records from a JSON file",
    )
    input_group.add_argument(
        "--from-db",
        action="store_true",
        help="Load IndustryGroup records from Memgraph",
    )

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7687)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--output-cypherl", type=str)
    parser.add_argument("--save-to-db", action="store_true")

    args = parser.parse_args()

    if args.from_json:
        print(f"Loading industry groups from {args.from_json}")
        industry_groups = load_industry_groups_from_json(args.from_json)
    else:
        print(
            f"Loading industry groups from Memgraph at {args.host}:{args.port}"
        )
        industry_groups = load_industry_groups_from_memgraph(
            args.host,
            args.port,
        )

    print(f"Loaded {len(industry_groups)} industry groups")
    industry_groups = generate_embeddings(
        industry_groups,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    if args.output_json:
        save_embeddings_to_json(industry_groups, args.output_json)

    if args.output_cypherl:
        save_embeddings_to_cypherl(industry_groups, args.output_cypherl)

    if args.save_to_db:
        save_embeddings_to_memgraph(industry_groups, args.host, args.port)

    if not any([args.output_json, args.output_cypherl, args.save_to_db]):
        save_embeddings_to_json(industry_groups, "output/sic_embeddings.json")
        save_embeddings_to_cypherl(
            industry_groups,
            "output/sic_embeddings.cypherl",
        )


if __name__ == "__main__":
    main()
