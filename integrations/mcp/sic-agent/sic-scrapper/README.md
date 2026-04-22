# SIC Scraper

Scrapes the SIC (Standard Industrial Classification) hierarchy from OSHA and
generates Cypher queries for Memgraph import.

## SIC Hierarchy

```text
Division (A-J) -> Major Group (2-digit) -> Industry Group (3-digit) -> Industry (4-digit)
```

## Usage

```bash
uv sync

# Full scrape
uv run main.py

# Quick scrape (skip industry details)
uv run main.py --no-industry-details
```

## Output

- `output/sic_data.json` - Complete hierarchy
- `output/sic_import.cypherl` - Cypher import queries
- `output/sic_embeddings.json` - Generated IndustryGroup embeddings
- `output/sic_embeddings.cypherl` - Embedding update queries

## Import to Memgraph

```bash
mgconsole < output/sic_import.cypherl
mgconsole < output/sic_embeddings.cypherl
mgconsole < output/sic_vector_index.cypherl
```

## Generate Embeddings

```bash
uv run embeddings.py --from-json output/sic_data.json
```

## Data Source

https://www.osha.gov/data/sic-manual
