import openai
import streamlit as st
from sentence_transformers import SentenceTransformer
import asyncio
from openai import AsyncOpenAI
import neo4j

# Replace 'your-openai-api-key' with your actual OpenAI API key.
openai.api_key = ""


def embed_question(key_information, model):
    return model.encode(key_information)


def question_prompt(question):
    prompt = f"""
    You are an AI language model. I will provide you with a question. 
    Extract the key information from the question. The key information is important information that is required to answer the question.

    Question: {question}

    The output format should be like this: 
    Key Information: [key information 1], [key information 2], ...
    """
    return prompt


def RAG_prompt(question, relevance_expansion_data):
    prompt = f"""
    You are an AI language model. I will provide you with a question and a set of data obtained through a relevance expansion process in a graph database. The relevance expansion process finds nodes connected to a target node within a specified number of hops and includes the relationships between these nodes.

    Question: {question}

    Relevance Expansion Data:
    {relevance_expansion_data}

    Based on the provided data, please answer the question, make sure to base your answers only based on the provided data. Add a context on what data did you base your answer on.
    """
    return prompt


def run_graphrag_query(driver, question, question_embedding, hops):
    pivot_node = find_most_similar_node(driver, question_embedding)
    relevant_data = get_relevant_data(driver, pivot_node, hops)
    prompt = RAG_prompt(question, relevant_data)
    response = asyncio.run(get_response(client, prompt))
    return response


def find_most_similar_node(driver, question_embedding):

    with driver.session() as session:
        # Perform the vector search on all nodes based on the question embedding
        result = session.run(
            f"CALL vector_search.search('got_index', 10, {question_embedding.tolist()}) YIELD * RETURN *;"
        )
        nodes_data = []

        # Retrieve all similar nodes and print them
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

        # Return the most similar node
        return nodes_data[0] if nodes_data else None


def get_relevant_data(driver, node, hops):
    with driver.session() as session:
        # Retrieve the paths from the node to other nodes that are 'hops' away
        query = f"MATCH path=((n)-[r*..{hops}]-(m)) WHERE id(n) = {node['id']} RETURN path LIMIT 100"
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

        # Return all paths
        return paths


async def get_response(client, prompt):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Streamlit App
st.title("GraphRAG query with Memgraph")
st.write("Ask a question related to your dataset.")

# Input box for user query
user_input = st.text_input("You:", placeholder="Enter your message here...")
emedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
# Replace 'api-key' with your actual OpenAI API key.
client = AsyncOpenAI(api_key="")
driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

# Process user input and display the response
if user_input:
    prompt = question_prompt(user_input)
    key_information = (
        asyncio.run(get_response(client, prompt)).split("Key Information: ")[1].strip()
    )
    st.write(
        f"Agent: Here is the key information extracted from the question: {key_information}"
    )
    question_embedding = embed_question(key_information, emedding_model)
    response = run_graphrag_query(driver, user_input, question_embedding, hops=2)
    st.write(f"Agent: {response}")
