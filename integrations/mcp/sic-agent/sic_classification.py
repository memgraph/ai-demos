"""SIC Classification MCP Server.

This server provides SIC (Standard Industrial Classification) code lookup
capabilities using vector search in Memgraph.

The server uses:
- Vector search to find relevant IndustryGroup nodes based on user description
- Node neighborhood exploration to gather context (Industries, MajorGroups)
- LLM sampling to determine the best matching SIC code
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from fastmcp import FastMCP
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable


def logger_init(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with a consistent configuration."""
    configured_logger = logging.getLogger(name)
    if not configured_logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        configured_logger.addHandler(handler)
        configured_logger.setLevel(level)
    return configured_logger


class MemgraphClient:
    """Minimal Memgraph client used by the SIC demo."""

    DEFAULT_USER_AGENT = "mcp-memgraph-sic"

    def __init__(
        self,
        url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        url = url or os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687")
        username = username or os.environ.get("MEMGRAPH_USER", "")
        password = password or os.environ.get("MEMGRAPH_PASSWORD", "")
        database = database or os.environ.get("MEMGRAPH_DATABASE", "memgraph")

        self.driver = GraphDatabase.driver(
            url,
            auth=(username, password),
            user_agent=user_agent or self.DEFAULT_USER_AGENT,
        )
        self.database = database

        try:
            self.driver.verify_connectivity()
        except ServiceUnavailable as error:
            raise ValueError(
                "Could not connect to Memgraph database. "
                f"Please ensure the URL '{url}' is correct"
            ) from error
        except AuthError as error:
            raise ValueError(
                "Could not connect to Memgraph database. "
                f"Authentication failed for user '{username}'"
            ) from error

    def query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as dictionaries."""
        parameters = params or {}
        try:
            data, _, _ = self.driver.execute_query(
                query,
                parameters_=parameters,
                database_=self.database,
            )
            return [record.data() for record in data]
        except Neo4jError as error:
            if not (
                (
                    (
                        error.code
                        == "Neo.DatabaseError.Statement.ExecutionFailed"
                        or error.code
                        == (
                            "Neo.DatabaseError.Transaction."
                            "TransactionStartFailed"
                        )
                    )
                    and "in an implicit transaction" in error.message
                )
                or (
                    error.code == "Neo.ClientError.Statement.SemanticError"
                    and (
                        "in an open transaction is not possible"
                        in error.message
                        or "tried to execute in an explicit transaction"
                        in error.message
                    )
                )
                or (
                    error.code
                    == "Memgraph.ClientError.MemgraphError.MemgraphError"
                    and "in multicommand transactions" in error.message
                )
                or (
                    error.code
                    == "Memgraph.ClientError.MemgraphError.MemgraphError"
                    and "SchemaInfo disabled" in error.message
                )
            ):
                raise

        with self.driver.session(database=self.database) as session:
            data = session.run(query, parameters)
            return [record.data() for record in data]


logger = logger_init("mcp-memgraph-sic")
mcp = FastMCP("mcp-memgraph-sic")

MEMGRAPH_URL = os.environ.get("MEMGRAPH_URL", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USER", "")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")
MEMGRAPH_DATABASE = os.environ.get("MEMGRAPH_DATABASE", "memgraph")

SIC_VECTOR_INDEX = os.environ.get(
    "SIC_VECTOR_INDEX", "sic_industry_group_embedding"
)
EMBEDDING_MODEL = os.environ.get("SIC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_db() -> MemgraphClient:
    """Initialize the Memgraph client on first use."""
    logger.info(
        "Connecting to Memgraph db '%s' at %s",
        MEMGRAPH_DATABASE,
        MEMGRAPH_URL,
    )
    return MemgraphClient(
        url=MEMGRAPH_URL,
        username=MEMGRAPH_USERNAME,
        password=MEMGRAPH_PASSWORD,
        database=MEMGRAPH_DATABASE,
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    except ImportError:
        logger.error(
            "sentence_transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        raise
    return embedding_model


def get_embedding_from_text(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def get_node_context(
    node_id: int,
    max_distance: int = 1,
) -> list[dict[str, Any]]:
    """Get the neighborhood context around a node."""
    query = (
        f"MATCH (n)-[r*..{max_distance}]-(m) WHERE id(n) = {int(node_id)} "
        "RETURN DISTINCT m LIMIT 50"
    )
    try:
        results = get_db().query(query)
        return [dict(record["m"]) for record in results if "m" in record]
    except (Neo4jError, KeyError, TypeError, ValueError) as error:
        logger.error("Failed to get node neighborhood: %s", str(error))
        return []


def perform_vector_search(
    query_vector: list[float], limit: int = 3
) -> list[dict[str, Any]]:
    """Perform vector search on the SIC index."""
    query = (
        f"CALL vector_search.search(\"{SIC_VECTOR_INDEX}\", "
        f"{limit}, $query_vector) "
        "YIELD node, distance RETURN node, distance;"
    )
    try:
        results = get_db().query(query, {"query_vector": query_vector})
        records = []
        for record in results:
            node = dict(record["node"])
            properties = {
                key: value for key, value in node.items() if key != "embedding"
            }
            records.append(
                {
                    "properties": properties,
                    "distance": record["distance"],
                }
            )
        return records
    except (Neo4jError, KeyError, TypeError, ValueError) as error:
        logger.error("Vector search failed: %s", str(error))
        return []


def get_node_id_by_code(code: str, label: str = "IndustryGroup") -> int | None:
    """Get the internal node ID by the SIC code."""
    try:
        query = f"MATCH (n:{label} {{code: $code}}) RETURN id(n) AS node_id"
        results = get_db().query(query, {"code": code})
        if results:
            return results[0]["node_id"]
        return None
    except (Neo4jError, KeyError, TypeError, ValueError) as error:
        logger.error("Failed to get node ID: %s", str(error))
        return None


async def generate_clarifying_facts(
    prompt: str,
    candidates: list[dict[str, Any]],
    ctx: Any,
) -> dict[str, Any]:
    """Generate clarifying statements for ambiguous classifications."""
    candidates_text = _format_candidates_for_prompt(candidates)

    analysis_prompt = f"""You are an expert in SIC
(Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

Based on vector similarity search, here are the top candidate SIC
classifications:
{candidates_text}

The description is ambiguous or could match multiple SIC codes.
Generate exactly 3 clarifying statements that would help
distinguish between the possible classifications.

Each fact should be a simple statement that the user can confirm
or deny about their business.

Return your response as JSON with this structure:
{{
    "fact_1": "Your business primarily involves [specific activity A]",
    "fact_2": "Your business primarily involves [specific activity B]",
    "fact_3": "Your business primarily involves [specific activity C]",
    "reasoning": "Why these facts distinguish between candidates"
}}

IMPORTANT:
- Each fact should clearly map to one of the candidate SIC codes
- Facts should be mutually exclusive where possible
- Use simple, clear language the user can easily understand
"""

    try:
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a SIC classification expert. "
                "Generate clarifying facts "
                "to help distinguish between possible SIC codes. "
                "Return only valid JSON, no additional text or markdown."
            ),
            temperature=0.3,
            max_tokens=500,
        )

        response_text = _extract_response_text(response)
        return json.loads(response_text)
    except json.JSONDecodeError as error:
        logger.error(
            "Failed to parse clarifying facts response: %s",
            str(error),
        )
        return {"error": "Failed to generate clarifying facts"}
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.error("Clarifying facts generation failed: %s", str(error))
        return {"error": f"Failed to generate facts: {str(error)}"}


async def analyze_with_user_selection(
    prompt: str,
    candidates: list[dict[str, Any]],
    selected_fact: str,
    ctx: Any,
) -> dict[str, Any]:
    """Perform the final SIC selection after clarification."""
    candidates_text = _format_candidates_for_prompt(candidates)

    analysis_prompt = f"""You are an expert in SIC
(Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

The user has confirmed the following fact about their business:
"{selected_fact}"

Based on vector similarity search, here are the candidate SIC classifications:
{candidates_text}

Given the user's confirmation, select the BEST matching SIC code.

Return your response as JSON with this structure:
{{
    "selected_code": "XXXX",
    "selected_name": "Name of the selected classification",
    "confidence": "high",
        "explanation": "Why this code was selected based on the
        confirmed fact"
}}

IMPORTANT:
- The confidence should now be "high" since the user has clarified
    their business
- Use the most specific code possible (4-digit Industry code if available)
"""

    try:
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a SIC classification expert. "
                "Select the best SIC code "
                "based on the user's confirmed business activity. "
                "Return only valid JSON, no additional text or markdown."
            ),
            temperature=0.1,
            max_tokens=500,
        )

        response_text = _extract_response_text(response)
        return json.loads(response_text)
    except json.JSONDecodeError as error:
        logger.error("Failed to parse final analysis response: %s", str(error))
        return {"error": "Failed to parse final classification"}
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.error("Final analysis failed: %s", str(error))
        return {"error": f"Final classification failed: {str(error)}"}


def _format_candidates_for_prompt(candidates: list[dict[str, Any]]) -> str:
    """Format candidate SIC entries for LLM prompts."""
    candidates_text = ""
    for index, candidate in enumerate(candidates, 1):
        candidates_text += f"\n--- Candidate {index} ---\n"
        candidates_text += (
            f"Industry Group Code: {candidate.get('code', 'N/A')}\n"
        )
        candidates_text += (
            f"Industry Group Name: {candidate.get('name', 'N/A')}\n"
        )
        candidates_text += f"Context: {candidate.get('context_text', 'N/A')}\n"

        if "related_industries" in candidate:
            candidates_text += "Related Industries:\n"
            for industry in candidate["related_industries"][:5]:
                ind_code = industry.get("code", "")
                ind_name = industry.get("name", "")
                ind_desc = industry.get("description", "")
                candidates_text += f"  - {ind_code}: {ind_name} - {ind_desc}\n"

        if "major_group" in candidate and candidate["major_group"]:
            major_group = candidate["major_group"]
            mg_code = major_group.get("code", "")
            mg_name = major_group.get("name", "")
            mg_desc = major_group.get("description", "")
            candidates_text += (
                f"Major Group: {mg_code}: {mg_name} - {mg_desc}\n"
            )

    return candidates_text


def _extract_response_text(response: Any) -> str:
    """Extract text from FastMCP sampling responses."""
    if isinstance(response, str):
        response_text = response.strip()
    elif hasattr(response, "text"):
        response_text = response.text.strip()
    else:
        response_text = str(response).strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(
            line
            for line in lines
            if not line.strip().startswith("```")
            and line.strip().lower() != "json"
        ).strip()

    return response_text


async def analyze_and_select_sic_code(
    prompt: str,
    candidates: list[dict[str, Any]],
    ctx: Any,
) -> dict[str, Any]:
    """Use LLM sampling to analyze candidates and select the best SIC code."""
    candidates_text = _format_candidates_for_prompt(candidates)

    analysis_prompt = f"""You are an expert in SIC
(Standard Industrial Classification) codes.

A user has described their business activity as follows:
"{prompt}"

Based on vector similarity search, here are the top candidate SIC
classifications:
{candidates_text}

Your task:
1. Analyze how well each candidate matches the user's business description
2. Select the BEST matching SIC code (4-digit code from Industries
    if specific match, or Industry Group code if more general)
3. Determine if you are confident in this match

Return your response as JSON with this structure:
{{
    "selected_code": "XXXX",
    "selected_name": "Name of the selected classification",
    "confidence": "high" or "low",
    "explanation": "Detailed explanation of why this code was selected"
}}

CONFIDENCE RULES:
- "high": The user's description clearly and unambiguously matches one SIC code
- "low": The description is vague, ambiguous, or could match multiple SIC codes

IMPORTANT:
- Use the most specific code possible (4-digit Industry code if available)
- Be conservative - if there's any ambiguity, use "low" confidence
- Consider the full context including related industries and major groups
"""

    try:
        response = await ctx.sample(
            messages=analysis_prompt,
            system_prompt=(
                "You are a SIC classification expert. "
                "Analyze business descriptions "
                "and match them to the most appropriate SIC code. "
                "Return only valid JSON, no additional text or markdown."
            ),
            temperature=0.2,
            max_tokens=500,
        )

        response_text = _extract_response_text(response)
        result = json.loads(response_text)
        confidence = result.get("confidence", "low").lower()
        result["confidence"] = "high" if confidence == "high" else "low"
        return result
    except json.JSONDecodeError as error:
        logger.error("Failed to parse LLM response: %s", str(error))
        return {
            "error": "Failed to parse classification response",
            "raw_response": (
                response_text if "response_text" in locals() else None
            ),
        }
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.error("Classification analysis failed: %s", str(error))
        return {"error": f"Classification failed: {str(error)}"}


@mcp.tool()
async def get_sic(prompt: str, ctx: Any) -> dict[str, Any]:
    """Get the SIC code that best matches a business description."""
    logger.info("get_sic called with prompt: %s", prompt)

    logger.info("Generating embedding for prompt...")
    query_embedding = get_embedding_from_text(prompt)

    logger.info("Performing vector search...")
    search_results = perform_vector_search(query_embedding, limit=3)

    logger.info("Vector search returned %d results", len(search_results))
    for index, result in enumerate(search_results):
        properties = result.get("properties", {})
        logger.info(
            "  Candidate %d: code=%s, name=%s, distance=%s",
            index + 1,
            properties.get("code", "N/A"),
            properties.get("name", "N/A"),
            result.get("distance", "N/A"),
        )

    if not search_results:
        return {
            "error": "No matching SIC classifications found",
            "prompt": prompt,
        }

    logger.info("Gathering context for %d candidates...", len(search_results))
    candidates = []
    for result in search_results:
        properties = result.get("properties", {})
        code = properties.get("code", "")

        candidate = {
            "code": code,
            "name": properties.get("name", ""),
            "context_text": properties.get("context_text", ""),
            "distance": result.get("distance", 0),
            "related_industries": [],
            "major_group": None,
        }

        node_id = get_node_id_by_code(code)
        if node_id is not None:
            neighborhood = get_node_context(node_id, max_distance=1)
            for neighbor in neighborhood:
                if "examples" in neighbor:
                    candidate["related_industries"].append(
                        {
                            "code": neighbor.get("code", ""),
                            "name": neighbor.get("name", ""),
                            "description": neighbor.get("description", ""),
                            "examples": neighbor.get("examples", []),
                        }
                    )
                elif "description" in neighbor and "embedding" not in neighbor:
                    candidate["major_group"] = {
                        "code": neighbor.get("code", ""),
                        "name": neighbor.get("name", ""),
                        "description": neighbor.get("description", ""),
                    }

        candidates.append(candidate)

        major_group_code = None
        if candidate["major_group"]:
            major_group_code = candidate["major_group"].get("code")
        logger.info(
            "  Context for %s: %d related industries, major_group=%s",
            code,
            len(candidate["related_industries"]),
            major_group_code,
        )

    logger.info("Performing initial analysis with LLM...")
    analysis_result = await analyze_and_select_sic_code(
        prompt,
        candidates,
        ctx,
    )

    if "error" in analysis_result:
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        return analysis_result

    if analysis_result.get("confidence") == "high":
        logger.info("High confidence result, returning immediately")
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        return analysis_result

    logger.info("Low confidence, generating clarifying facts...")
    facts_result = await generate_clarifying_facts(prompt, candidates, ctx)
    if "error" in facts_result:
        logger.warning("Failed to generate facts, returning initial result")
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        analysis_result["clarification_failed"] = True
        return analysis_result

    fact_1 = facts_result.get("fact_1", "Option 1")
    fact_2 = facts_result.get("fact_2", "Option 2")
    fact_3 = facts_result.get("fact_3", "Option 3")

    elicit_message = (
        "To better classify your business, please select the statement "
        "that best describes your primary activity:\n\n"
        f"1. {fact_1}\n\n"
        f"2. {fact_2}\n\n"
        f"3. {fact_3}"
    )

    logger.info("Eliciting user selection...")

    try:
        elicit_result = await ctx.elicit(
            message=elicit_message,
            response_type=["1", "2", "3"],
        )

        if elicit_result.action == "accept":
            selected_option = elicit_result.data
            logger.info("User selected option: %s", selected_option)

            if selected_option == "1":
                selected_fact = fact_1
            elif selected_option == "2":
                selected_fact = fact_2
            else:
                selected_fact = fact_3

            logger.info("Performing final analysis with user selection...")
            final_result = await analyze_with_user_selection(
                prompt,
                candidates,
                selected_fact,
                ctx,
            )
            final_result["candidates"] = candidates
            final_result["prompt"] = prompt
            final_result["user_clarification"] = selected_fact
            return final_result

        if elicit_result.action == "decline":
            logger.info("User declined clarification")
            analysis_result["candidates"] = candidates
            analysis_result["prompt"] = prompt
            analysis_result["user_declined_clarification"] = True
            return analysis_result

        logger.info("User cancelled")
        return {
            "status": "cancelled",
            "message": "Classification cancelled by user",
            "prompt": prompt,
        }
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.error("Elicitation failed: %s", str(error))
        analysis_result["candidates"] = candidates
        analysis_result["prompt"] = prompt
        analysis_result["elicitation_error"] = str(error)
        return analysis_result


logger.info("SIC Classification MCP server initialized")
logger.info("Available tools: get_sic")

if __name__ == "__main__":
    mcp.run()
