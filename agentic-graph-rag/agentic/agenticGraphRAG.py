from openai import OpenAI
from sentence_transformers import SentenceTransformer
import streamlit as st
import neo4j
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv
from typing import Dict, List, Any
import os
import json
import logging
import tiktoken

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO (@antejavor): Figure out how to handle data flow in the pipelines, represent success and failure

MODEL = { 
        "name" : "gpt-4o-2024-08-06", 
        "context_window": 128000 
        } 

# Types of questions that are possible:
# - Specific Data Retrieval (retrieval): Get me the age of a person with the name "John".
# - Graph Structure Insights (structure): Does John have a job?
# - Path Finding Between Entities/Relations (path): Is John a friend of Mary? If not, what is the shortest path between them?
# - Database Statistical and Count Queries (statistical): How many people have a job? How many nodes are there in the graph?
# - Global Insights and Trends (global): What is the most important node in the graph?

# Define the list of possible questions types in list
QUESTION_TYPES = [
    "Retrieval",
    "Structure",
    "Statistical",
    "Global",
]

class ToolResponse():
    def __init__(self, status=False, results=""):
        self.status = status
        self.results = results
    
    def __str__(self):
        return f"Status: {self.status}, Results: {self.results}"

    def set_status(self, status: bool):
        self.status = status
        return self  

    def set_results(self, results: str):
        self.results = results
        return self  

# Agent generation of a Cypher question
class CypherQuery(BaseModel):
    query: str

# Agent response for tool selection
class ToolSelection(BaseModel):
    first_pick: str
    second_pick: str

# Agent generation for number of similar nodes and number of hops
class StructureQuestionData(BaseModel):
    number_of_similar_nodes: int
    number_of_hops: int

# Agent generation for number of nodes in the PageRank
class PageRankNodes(BaseModel):
    number_of_nodes: int


# Agent reponse to the user question
class QuestionType(BaseModel):
    type: str
    explanation: str


class Community(BaseModel):
    summary: str    


CLASSIFY_QUESTION_PROMPT = """
Classify the following user question into query type

    Query Types:
    - Retrieval
    - Structure 
    - Global
    - Database

    Each type of question has different characteristics.
    - Retrieval: Direct Lookups, specific and well-defined. The query seeks information about specific entities (nodes or relationships). 
    - Structure: Exploratory, the query seeks information about the structure of the graph, close relationships between entities, or properties of nodes.
    - Global: The query seeks context about the entire graph, community, such as the most important node or global trends in graph. 
    - Database: The query seeks statistical information about the database, such as index information, node count, or relationship count, config etc.

    Example of a questions for each type:
    - Retrieval: How old is a person with the name "John"? 
    - Structure: Does John have a job? Is John a friend of Mary? Are there any people who are friends with John?
    - Globals: What is the most important node in the graph? 
    - Database: What indexes does Memgraph have?

    In the explanation, provide a brief description of the type of question, and why you classified it as such. 

    The question is in <Question> </Question> format.

"""


# Classify the type of the question
def classify_the_question(openai_client, user_question: str) -> Dict:

    user_question = f"<Question>{user_question}</Question>"
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL["name"],
        messages=[
            {"role": "developer", "content": CLASSIFY_QUESTION_PROMPT},
            {"role": "user", "content": user_question},
        ],
        response_format=QuestionType,
    )

    return completion.choices[0].message.parsed




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


# Tool used to run text_to_Cypher
# TODO (@antejavor): Reminder for the examples of case-sensitivity and feedback loops 
def text_to_Cypher(db_client, openai_client, user_question) -> Dict:
    logger.info("Running text_to_cypher tool")

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
    - Use Memgraph MAGE procedures instead of Neo4j APOC procedures.

    With all the above information and instructions, generate Cypher query
    for the user question.
    """

    encoding = tiktoken.get_encoding("cl100k_base")
    token_count_user = len(encoding.encode(prompt_user))
    token_count_developer = len(encoding.encode(prompt_developer))
    token_count = token_count_user + token_count_developer
    logger.info(f"Token count on prompt : {token_count}")

    prompt_chain = [
            {"role": "developer", "content": prompt_developer},
            {"role": "user", "content": prompt_user},
    ]

    tool_response = ToolResponse()

    query = ""
    if token_count <= MODEL["context_window"]:
        query = generate_cypher_query(openai_client, prompt_chain)
    else:
        return tool_response.set_status(False).set_results("Token count exceeded the limit.")

    logger.info("### Cypher Query:")
    logger.info(query)
    
    res = []
    with db_client.session() as session:
        for _ in range(3):  # Try correction process up to 3 times
            try:
                results = session.run(query)
                if not results.peek():
                    raise ValueError(
                        "The query did not return any results. There is a possible issue with the query "
                        "labels and parameters if you are matching strings consider matching them in the case-insensitive way."
                    )
                for record in results:
                    res.append(record)

                return tool_response.set_status(True).set_results(res)

            except (ValueError, Exception) as e:
                error_type = "ValueError" if isinstance(e, ValueError) else "Error"
                logger.error(f"{error_type} in running the query:")
                logger.error(e)
                error_message = str(e)

                prompt_correction = f"""
                The following Cypher query generated a {error_type}:
                Query: {query}
                Error: {error_message}
                Question: {user_question}

                Please correct the Cypher query based on the error, schema and question.
                """
                prompt_chain.append({"role": "assistant", "content": query})
                prompt_chain.append({"role": "developer", "content": prompt_correction})

                query = generate_cypher_query(openai_client, prompt_chain)
                logger.info("### Corrected Cypher Query:")
                logger.info(query)

        return tool_response.set_status(False).set_results("Error in running the query.")

    

def generate_cypher_query(openai_client, prompt_messages):
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL["name"],
        messages=prompt_messages,
        response_format=CypherQuery,
    )
    return completion.choices[0].message.parsed.query



# Schema tool
def schema_tool(db_client) -> ToolResponse:
    return ToolResponse(True, get_schema_string(db_client))


# Config tool
def config_tool(db_client) -> ToolResponse:
    try:
        with db_client.session() as session:
            config = session.run("SHOW CONFIG")
            config_str = "Configurations:\n"
            for record in config:
                config_str += f"Name: {record['name']} | Default Value: {record['default_value']} | Current Value: {record['current_value']} | Description: {record['description']}\n"
            return ToolResponse(True, config_str)
    except Exception as e:
        logger.error("Error in running the Config tool query.")
        return ToolResponse(False, "Error in running the Config tool query.")

def page_rank_choice(openai_client, user_question) -> Dict:
    question = f"<Question>{user_question}</Question>"
    prompt = f"""
    Based on the provided question, try to guess how many nodes should be returned from the PageRank in the assesment. 
    The question is in <Question> </Question> format.
    """
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL["name"],
        messages=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": question},
        ],
        response_format=PageRankNodes,
    )

    return completion.choices[0].message.parsed

# Page Rank tool
def page_rank_tool(db_client, openai_client, user_question) -> ToolResponse:

    prompt_developer = f"""
    Based on the provided question, try to guess how many nodes should be returned from the PageRank in the assesment. 
    The question is in <Question> </Question> format.
    """

    messages = [
        {"role": "developer", "content": prompt_developer},
        {"role": "user", "content": "<Question>" + user_question + "</Question>"},
    ]

    choice = page_rank_choice(openai_client, user_question)

    logger.info("Running the PageRank tool")
    logger.info(f"Number of nodes: {choice.number_of_nodes}")

    with db_client.session() as session:
        try:
            result = session.run(f"CALL pagerank.get() YIELD node, rank RETURN node, rank LIMIT {choice.number_of_nodes};")
            result_str = ""
            for record in result:
                node = record["node"]
                properties = {k: v for k, v in node.items() if k != "embedding"}
                result_str += f"Node: {properties}, Rank: {record['rank']}\n"
            
            logger.info("Page rank successful") 
            logger.info(result_str)
            return ToolResponse(True, result_str)
        except Exception as e:
            logger.error("Error in running the PageRank tool query.")
            return ToolResponse(False, "Error in running the PageRank tool query.")


def community_tool(db_client) -> ToolResponse:
    try:
        with db_client.session() as session:
            result = session.run("MATCH (n:Community) RETURN n.id, n.summary;")
            result_str = ""
            for record in result:
                result_str += f"Community ID: {record['n.id']}, Summary: {record['n.summary']}\n"
            return ToolResponse(True, result_str)
    except Exception as e:
        logger.error("Error in running the Community tool query.")
        return ToolResponse(False, "Error in running the Community tool query.")


def community_prompt(openai_client, community_string) -> Dict:
    prompt = f"Summarize the following community information into 5 to 10 sentences, you will get the community string in the <Community> </Community> format"
    prompt_community= f"<Community>{community_string}</Community>"

    completion = openai_client.beta.chat.completions.parse(
        model=MODEL["name"],
        messages=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": prompt_community},
        ],
        response_format=Community,
    )

    return completion.choices[0].message.parsed


def precompute_community_summary(db_client, openai_client) -> Dict:

    number_of_communities = 0
    try:
        with db_client.session() as session:
            result = session.run("""
            CALL community_detection.get()
            YIELD node, community_id 
            SET node.community_id = community_id;
            """
            )
            result = session.run("""
            MATCH (n)
            RETURN count(distinct n.community_id) as community_count;
            """
            )
            for record in result:
                number_of_communities = record['community_count']
                print(f"Number of communities: {record['community_count']}")
    except Exception as e:
        logger.error("Error in running the community detection query.")
        return False; 
    
    try:
        with db_client.session() as session:
            communities = []
            for i in range(0, number_of_communities):
                community_string = ""
                community_id = 0
                result = session.run(f"""
                MATCH (start), (end) 
                WHERE start.community_id = {i} AND end.community_id = {i} AND id(start) < id(end)
                MATCH p = (start)-[*..1]-(end)
                RETURN p; 
                """)
                for record in result:
                    path = record['p']
                    for rel in path.relationships:
                        start_node = rel.start_node
                        end_node = rel.end_node
                        start_node_properties = {k: v for k, v in start_node.items() if k != 'embedding'}
                        end_node_properties = {k: v for k, v in end_node.items() if k != 'embedding'}
                        community_string += f"({start_node_properties})-[:{rel.type}]->({end_node_properties})\n"
                        community_id = i
                communities.append({"id": community_id, "data": community_string})
    except Exception as e:
        logger.error("Error in running the community detection query.")
        return False;
        
    logger.info("Total number of communities:")
    logger.info(number_of_communities)
    community_summary = []
    for community in communities:
        community_id = community['id']
        community_string = community['data']
        try:
            logging.info(f"Generating summary for community {community_id}")
            prompt = community_prompt(openai_client, community_string)
            community_summary.append({"id": community_id, "summary": prompt.summary})
        except Exception as e:
            logger.error(f"Error in generating summary for community {community_id} and community string {community_string}")
            return False;

    try:
        with db_client.session() as session:
            for community in community_summary:
                community_id = community['id']
                summary = community['summary']
                session.run(
                    "CREATE (c:Community { id: $id, summary: $summary})",
                    summary=summary, 
                    id=community_id
                )
    except Exception as e:
        logger.error("Error in running the community detection query.")
        return False;
    
    return True



def decide_on_structure_parameters(openai_client, messages) -> Dict:
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL["name"],
        messages=messages,
        response_format=StructureQuestionData,
    )
    return completion.choices[0].message.parsed



def vector_relevance_expansion(db_client, openai_client, user_question) -> Dict:

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    question_embedding = model.encode(user_question)

    prompt_parameters = f"""
    You will get a question about the structure of the graph. The vector search
    will find the most similar node based on the question embedding an node
    embedding, and then return the data connected to the most similar nodes that
    are hops away. Your task is to find out how many nodes should vector search
    return and how many hops should be used to find the relevant data. If the
    question is about undefined number of node guess the intended number of
    nodes, by default consider 1. If the question is about undefined number of
    hops guess the intended number of hops, by default consider 1.
    """

    messages = [ 
        {"role": "developer", "content": prompt_parameters}, 
        {"role": "user", "content": user_question} 
        ]

    structure_parameters = decide_on_structure_parameters(openai_client, messages)

    logger.info("Structure parameters:")
    logger.info(structure_parameters)

    nodes = find_most_similar_nodes(db_client, user_question,  question_embedding, structure_parameters.number_of_similar_nodes)


    for node in nodes:
        logger.info("Most similar nodes:")
        logger.info(node)

    tool_response = ToolResponse()
    if nodes is None:
        return tool_response.set_status(False).set_results("No similar nodes found.")

    relevant_data = get_relevant_data(db_client, nodes, structure_parameters.number_of_hops)

    return tool_response.set_status(True).set_results(relevant_data)
    

def find_most_similar_nodes(db_client, user_question,  question_embedding, number_of_similar_nodes):
        
    with db_client.session() as session:
        result = session.run(
            f"CALL vector_search.search('index_name', {number_of_similar_nodes}, {question_embedding.tolist()}) YIELD * RETURN *;"
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

        return nodes_data if nodes_data else None


def get_relevant_data(db_client, nodes, hops):
    paths = []
    for node in nodes:
        with db_client.session() as session:
            query = (
                f"MATCH path=((n)-[r*..{hops}]-(m)) WHERE id(n) = {node['id']} RETURN path"
            )
            result = session.run(query)
            
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




def generate_final_response(openai_client, results, user_question: str):
    prompt = f"""
    Using the data and the user's original question, generate a final answer:
    User Question: "{user_question}"
    Data from the database: {results}

    Try to answer the user's question using just the the provided data..
    
    """
    completion = openai_client.chat.completions.create(
        model=MODEL["name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message


def index_setup(db_client):
    with db_client.session() as session:
        print("Creating the vector index...")
        session.run(
            """
            CREATE VECTOR INDEX index_name ON :Entity(embedding) WITH CONFIG {"dimension": 384, "capacity": 2000, "metric": "cos","resize_coefficient": 2};
            """
        )

def compute_node_embeddings(db_client):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    with db_client.session() as session:
        # Retrieve all nodes
        result = session.run("MATCH (n) RETURN n")
        print("Embedded data: ")
        for record in result:
            node = record["n"]
            # Check if the node already has an embedding
            if "embedding" in node:
                print("Embedding already exists")
                return

            # Combine node labels and properties into a single string
            node_data = (
                " ".join(node.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node.items())
            )
            print(node_data)
            # Compute the embedding for the node
            node_embedding = model.encode(node_data)

            # Store the embedding back into the node
            session.run(
                f"MATCH (n) WHERE id(n) = {node.element_id} SET n.embedding = {node_embedding.tolist()}"
            )

        session.run("MATCH (n) SET n:Entity")


@st.cache_resource()
def get_openai_client():
    return OpenAI()

@st.cache_resource()
def get_db_client():
    return neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

@st.cache_resource()
def preprocess_data(_db_client, _openai_client):
    # status = precompute_community_summary(db_client, openai_client)
    # if status:
    #     logger.info("Community summary precomputed.")
    # else:
    #     logger.error("Error in precomputing community summary.")
    #     logger.error("Community questions will fail")

    index_setup(db_client)
    compute_node_embeddings(db_client)
    return "Proccessing data completed"


def tool_selection_pipe(openai_client, user_question, question_type) -> Dict:
    question = f"<Question>{user_question}</Question>"
    question_type = f"<Type>{question_type}</Type>"
    prompt_developer = f"""

    Based on the question type and the user's question, select the most appropriate tool option and second option as a backup to answer the question:

    Retrival - direct lookups, specific and well-defined. The query seeks information about specific entities (nodes or relationships).
    Options: 
        - Cypher: A tool that generates a Cypher query based on the user's question and the database schema.
        - Vector Relevance Expansion: A tool that finds the most similar nodes based on the user's question and the database schema.
    Structure - exploratory, the query seeks information about the structure of the graph, close relationships between entities, or properties of nodes.
    Options:
        - Cypher: A tool that generates a Cypher query based on the user's question and the database schema.
        - Vector Relevance Expansion: A tool that finds the most similar nodes based on the user's question and the database schema.
        - Global - the query seeks context about the entire graph, community, such as the most important node or global trends in graph.
    Global - the query seeks context about the entire graph, community, such as the most important node or global trends in graph.
    Options:
        - PageRank: A tool that provides PageRank information about the graph and its nodes, it can help with identifying the most important nodes.
        - Community: A tool that provides communities information about the graph, it contains the summary of the community, and can help with global insights.
    Database - the query seeks statistical information about the database, such as index information, node count, or relationship count, config etc.
    Options:
        - Schema: A tool that provides schema information about the dataset and datatypes.
        - Config: A tool that provides configuration information about the database.
        - Cypher: A tool that generates a Cypher query based on the user's question and the database schema.

    The question is in <Question> </Question> format, and the type of the question is <Type> </Type>.

    """
    messages = [
        {"role": "developer", "content": prompt_developer},
        {"role": "user", "content": question + question_type},

    ]
    completion = openai_client.beta.chat.completions.parse(
        model=MODEL["name"],
        messages=messages,
        response_format=ToolSelection,
    )

    return completion.choices[0].message.parsed


def tool_execution(tool: str, db_client, openai_client, user_question) -> ToolResponse:
    """
    Simulates execution success/failure for demo purposes.
    Replace this with actual logic to execute each tool.
    """
    # For demo, let's assume "Text to Cypher" always succeeds
    if tool == "Cypher":
        return text_to_Cypher(db_client, openai_client, user_question)
    elif tool == "Vector Relevance Expansion":
        return vector_relevance_expansion(db_client, openai_client, user_question)
    elif tool == "PageRank":
        return page_rank_tool(db_client, openai_client, user_question)
    elif tool == "Community":
        return community_tool(db_client)
    elif tool == "Schema":
        return  schema_tool(db_client)
    elif tool == "Config":
        return config_tool(db_client)
    else:
        return ToolResponse(False, "Tool execution failed, tool not found.")


def execute_tool(tool: str, user_question: str, db_client,  openai_client ) -> ToolResponse:
    """
    Executes the given tool based on its name.
    Returns True if successful, False otherwise.
    """
    response = None
    try:
        logger.info(f"Trying tool: {tool}")
        # TODO: Implement actual execution logic for each tool
        # Simulating success/failure for now
        response = tool_execution(tool, db_client, openai_client, user_question)
        return response
    except Exception as e:
        logger.error(f"Error executing {tool}: {e}")
        return response




def main(db_client, openai_client):

    st.title("Agentic GraphRAG with Memgraph")

    # User input
    user_question = st.text_input("Enter your question about the dataset:", "")

    if st.button("Run GraphRAG Pipeline"):
        if user_question.strip():

            st.write("## Classifying Question type...")
            logger.info("Classifying Question type...")
            question_type = classify_the_question(openai_client, user_question)

            st.write("### Question Type:")
            st.write("*Type*: ", question_type.type)
            st.write("*Explanation*: ", question_type.explanation)

            st.write("## Running the tool selection...")
            
            tools = tool_selection_pipe(openai_client, user_question, question_type)

            st.write("### Tools selected:") 
            st.write("Tool 1 ", tools.first_pick)  
            st.write("Tool 2: ", tools.second_pick)

             # Try first pick
            response = execute_tool(tools.first_pick, user_question, db_client, openai_client)
            if response.status:
                logger.info(f"First pick '{tools.first_pick}' succeeded.")
            else:
                response = execute_tool(tools.second_pick, user_question, db_client, openai_client)
                if response.status:
                    logger.info(f"Second pick '{tools.second_pick}' succeeded.")
                else:
                    st.error("Tool failed to execute")
                    logger.error(f"Second pick '{tools.second_pick}' failed.")

            st.write("### Tool Execution Completed.")

            st.write("### Tool Response:")
            st.write(response.results)

            st.write("## Generating Final Response...")

            final_response = generate_final_response(
                openai_client, response.results, user_question
            )
            st.write("### Final Response:")
            st.write(final_response.content)
            st.write("## Agentic GraphRAG Pipeline Completed.")
                
        else:
            st.error("Please enter a question to proceed.")



if __name__ == "__main__":
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    openai_client = get_openai_client()

    db_client = get_db_client()

    preprocess_data(db_client, openai_client)

    main(db_client, openai_client)
