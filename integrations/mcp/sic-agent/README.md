# SIC Classification MCP Agent

This example ports the SIC classification agent from the Memgraph AI Toolkit
into this repository as a standalone FastMCP server.

The agent:
- embeds a free-form business description,
- performs vector search over `IndustryGroup` nodes in Memgraph,
- inspects nearby `Industry` and `MajorGroup` context,
- uses MCP sampling to select the best SIC code,
- asks a clarifying follow-up when the initial match is ambiguous.

## Files

- `sic_classification.py` runs the FastMCP server.
- `sic-scrapper/main.py` scrapes the OSHA SIC manual and generates import
  Cypher.
- `sic-scrapper/embeddings.py` generates embeddings for `IndustryGroup` nodes.
- `sic-scrapper/output/sic_vector_index.cypherl` creates the vector index used
  by the server.

## Quick Start

1. Generate SIC data and embedding updates:

```bash
cd sic-scrapper
uv sync
uv run main.py
uv run embeddings.py --from-json output/sic_data.json
```

2. Load the generated data into Memgraph:

```bash
mgconsole < output/sic_import.cypherl
mgconsole < output/sic_embeddings.cypherl
mgconsole < output/sic_vector_index.cypherl
```

3. Run the MCP server:

```bash
cd ..
uv sync
uv run python sic_classification.py
```

## Environment Variables

- `MEMGRAPH_URL` defaults to `bolt://localhost:7687`
- `MEMGRAPH_USER` defaults to an empty string
- `MEMGRAPH_PASSWORD` defaults to an empty string
- `MEMGRAPH_DATABASE` defaults to `memgraph`
- `SIC_VECTOR_INDEX` defaults to `sic_industry_group_embedding`
- `SIC_EMBEDDING_MODEL` defaults to `all-MiniLM-L6-v2`
