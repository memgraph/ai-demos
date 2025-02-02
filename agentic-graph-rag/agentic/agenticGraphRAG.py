from openai import OpenAI
from sentence_transformers import SentenceTransformer
import streamlit as st
import neo4j
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
import os
import json

from classify_question import get_classify_question_prompt


# TODO (@antejavor): Use OpenAI tiktoken for limits on the number of tokend used: https://github.com/openai/tiktoken?tab=readme-ov-file
# TODO (@antejavor): Make sure the code follows the chain of command: https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command
# TODO (@antejavor): Figure out path pipe for path questions
# TODO (@antejavor): Figure out how to handle data flow in the pipelines, represent success and failure

MODEL = "gpt-4o-2024-08-06"

# Types of questions that are possible:
# - Specific Data Retrieval (retrieval): Get me the age of a person with the name "John".
# - Graph Structure Insights (structure): Does John have a job?
# - Path Finding Between Entities/Relations (path): Is John a friend of Mary? If not, what is the shortest path between them?
# - Database Statistical and Count Queries (statistical): How many people have a job? How many nodes are there in the graph?
# - Global Insights and Trends (global): What is the most important node in the graph?


# Schema definition for tool selection
class QuestionType(BaseModel):
    type: str
    explanation: str


class CypherQuery(BaseModel):
    query: str


class ToolSelection(BaseModel):
    tool: str
    explanation: str


# Classify the type of the question
def classify_the_question(openai_client, user_question: str, config) -> Dict:
    classify_question_prompt = get_classify_question_prompt(config)
    user_question = f"<Question>{user_question}</Question>"
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "developer", "content": classify_question_prompt},
            {"role": "user", "content": user_question},
        ],
        response_format=QuestionType,
    )

    return completion.choices[0].message.parsed


def fetch_data(
    db_client, openai_client, user_question: str, question_type: Dict
) -> Dict:
    if question_type.type == "Retrieval":
        return run_retrival_pipe(db_client, openai_client, user_question)
    elif question_type.type == "Structure":
        return run_structure_pipe(db_client, openai_client, user_question)
    elif question_type.type == "Path":
        return {"answer": "Path finding is not implemented yet."}
    elif question_type.type == "Global":
        tool = tool_global_selection_pipe(openai_client, user_question)
        if tool.tool == "PageRank":
            return {"pagerank": page_rank_tool(db_client)}
        elif tool.tool == "Community":
            return {"community": community_tool(db_client)}
    elif question_type.type == "Database":
        tool = tool_database_selection_pipe(openai_client, user_question)
        if tool.tool == "Schema":
            return {"schema": schema_tool(db_client)}
        elif tool.tool == "Config":
            return {"config": config_tool(db_client)}
        elif tool.tool == "Cypher":
            return run_retrival_pipe(db_client, openai_client, user_question)
    else:
        return {"answer": "Problem with fetching the data."}


def run_other_pipe(openai_client, user_question, question_type) -> Dict:
    return {
        "question": user_question,
        "answer": "The question is classified as 'Other' and no further processing is done.",
        "type": question_type.type,
        "explaination": question_type.explanation,
    }


def get_schema_string(db_client) -> str:
    with db_client.session() as session:
        schema = session.run("SHOW SCHEMA INFO")
        schema_info = json.loads(schema.single().value())
        nodes = schema_info["nodes"]
        edges = schema_info["edges"]
        node_indexes = schema_info["node_indexes"]
        edge_indexes = schema_info["edge_indexes"]

        schema_str = "Nodes:\n"
        for node in nodes:
            properties = ", ".join(
                f"{prop['key']}: {', '.join(t['type'] for t in prop['types'])}"
                for prop in node["properties"]
            )
            schema_str += f"Labels: {node['labels']} | Properties: {properties}\n"

        schema_str += "\nEdges:\n"
        for edge in edges:
            properties = ", ".join(
                f"{prop['key']}: {', '.join(t['type'] for t in prop['types'])}"
                for prop in edge["properties"]
            )
            schema_str += f"Type: {edge['type']} | Start Node Labels: {edge['start_node_labels']} | End Node Labels: {edge['end_node_labels']} | Properties: {properties}\n"

        schema_str += "\nNode Indexes:\n"
        for index in node_indexes:
            schema_str += (
                f"Labels: {index['labels']} | Properties: {index['properties']}\n"
            )

        schema_str += "\nEdge Indexes:\n"
        for index in edge_indexes:
            schema_str += f"Type: {index['type']} | Properties: {index['properties']}\n"

        return schema_str


def run_retrival_pipe(db_client, openai_client, user_question) -> Dict:
    schema = get_schema_string(db_client)

    prompt_user = f"""
    User Question: "{user_question}"
    Schema: {schema}

    Based on schema and question, generate a Cypher query that directly corresponds to the user's intent.
    """

    prompt_developer = f"""
    Your task is to directly translate natural language
    inquiry into precise and executable Cypher query for Memgraph database.
    You will utilize a provided database schema to understand the structure,
    nodes and relationships within the Memgraph database.


    Rules:
    - Use provided node and relationship labels and property names from the
    schema which describes the database's structure. Upon receiving a user question, synthesize the
    schema to craft a precise Cypher query that directly corresponds to
    the user's intent.
    - Generate valid executable Cypher queries on top of Memgraph database.
    Any explanation, context, or additional information that is not a part
    of the Cypher query syntax should be omitted entirely.
    - Use Memgraph MAGE procedures instead of Neo4j APOC procedures.
    - Do not include any explanations or apologies in your responses.
    - Do not include any text except the generated Cypher statement.

    With all the above information and instructions, generate Cypher query
    for the user question.
    """

    query = generate_and_run_query(openai_client, prompt_developer, prompt_user)
    print("### Cypher Query:")
    print(query)

    res = []
    with db_client.session() as session:
        for _ in range(3):  # Try correction process up to 3 times
            try:
                results = session.run(query)
                for record in results:
                    print(record)
                    res.append(record)
                return {"query": query, "results": res}
            except Exception as e:
                print(e)
                error_message = str(e)
                correction_prompt = f"""
                The following Cypher query generated an error:
                Query: {query}
                Error: {error_message}
                Schema: {schema}
                Question: {user_question}

                Please correct the Cypher query based on the error, schema and question.
                """
                query = generate_and_run_query(
                    openai_client, prompt_developer, correction_prompt
                )
                print("### Corrected Cypher Query:")
                print(query)

        return {"answer": "Issue with the corrected query. " + query}


# Schema tool
def schema_tool(db_client) -> str:
    return get_schema_string(db_client)


# Config tool
def config_tool(db_client) -> str:
    with db_client.session() as session:
        config = session.run("SHOW CONFIG")
        config_str = "Configurations:\n"
        for record in config:
            config_str += f"Name: {record['name']} | Default Value: {record['default_value']} | Current Value: {record['current_value']} | Description: {record['description']}\n"
        return config_str


# Page Rank tool
def page_rank_tool(db_client) -> Dict:
    with db_client.session() as session:
        result = session.run(
            "CALL pagerank.get() YIELD node, rank RETURN node, rank LIMIT 10;"
        )
        result_str = ""
        for record in result:
            node = record["node"]
            properties = {k: v for k, v in node.items() if k != "embedding"}
            result_str += f"Node: {properties}, Rank: {record['rank']}\n"
        return result_str


def community_tool(db_client) -> Dict:
    with db_client.session() as session:
        result = session.run("MATCH (n:Community) RETURN n.id, n.summary;")
        result_str = ""
        for record in result:
            result_str += (
                f"Community ID: {record['n.id']}, Summary: {record['n.summary']}\n"
            )
        return result_str


def precompute_community_summary(db_client, openai_client) -> Dict:
    number_of_communities = 0
    with db_client.session() as session:
        result = session.run(
            """
        CALL community_detection.get()
        YIELD node, community_id 
        SET node.community_id = community_id;
        """
        )
        result = session.run(
            """
        MATCH (n)
        RETURN count(distinct n.community_id) as community_count;
        """
        )
        for record in result:
            number_of_communities = record["community_count"]
            print(f"Number of communities: {record['community_count']}")

    with db_client.session() as session:
        communities = []
        for i in range(0, number_of_communities):
            community_string = ""
            community_id = 0
            result = session.run(
                f"""
            MATCH (start), (end) 
            WHERE start.community_id = {i} AND end.community_id = {i} AND id(start) < id(end)
            MATCH p = (start)-[*..1]-(end)
            RETURN p; 
            """
            )
            for record in result:
                path = record["p"]
                for rel in path.relationships:
                    start_node = rel.start_node
                    end_node = rel.end_node
                    start_node_properties = {
                        k: v for k, v in start_node.items() if k != "embedding"
                    }
                    end_node_properties = {
                        k: v for k, v in end_node.items() if k != "embedding"
                    }
                    community_string += f"({start_node_properties})-[:{rel.type}]->({end_node_properties})\n"
                    community_id = i
            communities.append({"id": community_id, "data": community_string})

    print("Total number of communites: ", len(communities))
    community_summary = []
    for community in communities:
        community_id = community["id"]
        community_string = community["data"]
        # Generate summary using OpenAI LLM
        try:
            print("Creating a summary for the community with id:", community_id)
            prompt = f"Summarize the following community information into 5 sentences:\n{community_string}"
            completion = openai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "developer", "content": prompt}],
                temperature=0,
            )
            summary = completion.choices[0].message.content
            community_summary.append({"id": community_id, "summary": summary})
        except Exception as e:
            print(e)
            summary = "Summary could not be generated."
            print(summary)

    with db_client.session() as session:
        for community in community_summary:
            community_id = community["id"]
            summary = community["summary"]
            session.run(
                "CREATE (c:Community { id: $id, summary: $summary})",
                summary=summary,
                id=community_id,
            )

    return 0


def tool_global_selection_pipe(openai_client, user_question) -> Dict:
    question = f"<Question>{user_question}</Question>"

    prompt = f"""
    Select the tool that you see most fit for the asked question:
    {question}

    - PageRank: A tool that provides PageRank information about the graph and its nodes, it can help with identifying the most important nodes.
    - Community: A tool that provides communities information about the graph, it contains the summary of the community, and can help with global insights.
    """

    completion = openai_client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": question},
        ],
        response_format=ToolSelection,
    )

    return completion.choices[0].message.parsed


def tool_database_selection_pipe(openai_client, user_question) -> Dict:
    question = f"<Question>{user_question}</Question>"

    prompt = f"""
    Select the tool that you see most fit for the asked question:
    {question}

    - Schema: A tool that provides schema information about the dataset and d.
    - Config: A tool that provides configuration information about the database.
    - Cypher: A tool that provides a Cypher query to retrieve the requested data.
    """

    completion = openai_client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": question},
        ],
        response_format=ToolSelection,
    )

    return completion.choices[0].message.parsed


def generate_and_run_query(openai_client, prompt_developer, prompt_user):
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "developer", "content": prompt_developer},
            {"role": "user", "content": prompt_user},
        ],
        response_format=CypherQuery,
    )
    return completion.choices[0].message.parsed.query


def run_structure_pipe(db_client, openai_clien, user_question) -> Dict:
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    question_embedding = model.encode(user_question)
    node = find_most_similar_node(db_client, question_embedding)

    if node:
        print("The most similar node is:")
        print(node)

    relevant_data = get_relevant_data(db_client, node, hops=1)

    return relevant_data


def get_relevant_data(db_client, node, hops):
    with db_client.session() as session:
        query = (
            f"MATCH path=((n)-[r*..{hops}]-(m)) WHERE id(n) = {node['id']} RETURN path"
        )
        result = session.run(query)
        paths = []
        for record in result:
            path_data = []
            for segment in record["path"]:

                # Process start node without 'embedding' property
                start_node_data = {
                    k: v for k, v in segment.start_node.items() if k != "embedding"
                }

                # Process relationship data
                relationship_data = {
                    "type": segment.type,
                    "properties": segment.get("properties", {}),
                }

                # Process end node without 'embedding' property
                end_node_data = {
                    k: v for k, v in segment.end_node.items() if k != "embedding"
                }

                # Add to path_data as a tuple (start_node, relationship, end_node)
                path_data.append((start_node_data, relationship_data, end_node_data))

            paths.append(path_data)

        return paths


def find_most_similar_node(db_client, question_embedding):
    with db_client.session() as session:
        result = session.run(
            f"CALL vector_search.search('index_name', 10, {question_embedding.tolist()}) YIELD * RETURN *;"
        )
        nodes_data = []
        for record in result:
            node = record["node"]
            properties = {k: v for k, v in node.items() if k != "embedding"}
            node_data = {
                "distance": record["distance"],
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": properties,
            }
            nodes_data.append(node_data)
        print("All similar nodes:")
        for node in nodes_data:
            print(node)

        return nodes_data[0] if nodes_data else None


def generate_final_response(openai_client, results, user_question: str):
    prompt = f"""
    Using the data and the user's original question, generate a final answer:
    User Question: "{user_question}"
    Data from the database: {results}

    Try to answer the user's question using the provided data..
    
    """
    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message


@st.cache_resource()
def get_openai_client():
    return OpenAI()


@st.cache_resource()
def get_db_client():
    return neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))


@st.cache_resource()
def preprocess_data(_db_client, _openai_client, config):
    if config.get("graph-algorithms-enabled", False):
        precompute_community_summary(db_client, openai_client)

    return "Proccessing data completed"


def main(db_client, openai_client, config):
    st.title("Agentic GraphRAG with Memgraph")

    # User input
    user_question = st.text_input("Enter your question about the dataset:", "")

    if st.button("Run GraphRAG Pipeline"):
        if user_question.strip():
            st.write("### Classifying Question type...")
            question_type = classify_the_question(openai_client, user_question, config)
            st.json(question_type)

            if question_type.type == "Other":
                st.write("### Generating Response...")
                response = run_other_pipe(openai_client, user_question, question_type)
                st.json(response)
            else:
                st.write("### Fetching the data...")
                database_results = fetch_data(
                    db_client, openai_client, user_question, question_type
                )
                st.json(database_results)

                st.write("### Generating Final Response...")
                final_response = generate_final_response(
                    openai_client, database_results, user_question
                )
                st.json(final_response)
                st.write("### Pipeline Completed.")

        else:
            st.error("Please enter a question to proceed.")


def str_to_bool(value: str) -> bool:
    if str(value).strip().lower() == "true":
        return True
    if str(value).strip().lower() == "false":
        return False
    raise Exception(f"Unknown value for boolean {value}")


if __name__ == "__main__":
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = {
        "graph-algorithms-enabled": str_to_bool(
            os.environ.get("GRAPH_ALGORITHMS_ENABLED", "false")
        ),
        "vector-search-enabled": str_to_bool(
            os.environ.get("VECTOR_SEARCH_ENABLED", "false")
        ),
    }

    openai_client = get_openai_client()

    db_client = get_db_client()

    preprocess_data(db_client, openai_client, config)

    main(db_client, openai_client, config)
