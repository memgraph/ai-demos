from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import neo4j
import asyncio
from openai import AsyncOpenAI
import os
import json
from collections import Counter
from pathlib import Path
from time import sleep


def compute_tripplets_embeddings(driver, model):
    with driver.session() as session:
        # Retrieve all relationships
        result = session.run("MATCH (n:Person)-[r]->(m) RETURN n, r, m")
        print("Embedded data: ")

        for record in result:
            node1 = record["n"]
            relationship = record["r"]
            node2 = record["m"]
            # Check if the relationship already has an embedding
            if "embedding" in node1:
                print("Embedding already exists")
                return
            # Combine node labels and properties into a single string
            tripplet_data = (
                " ".join(node1.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node1.items())
                + " "
                + relationship.type
                + " "
                + " ".join(f"{k}: {v}" for k, v in relationship.items())
                + " "
                + " ".join(node2.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node2.items())
            )
            print(tripplet_data)
            # Compute the embedding for the tripplet
            tripplet_embedding = model.encode(tripplet_data)

            # Store the tripplet data on node1
            session.run(
                f"MATCH (n:Person) WHERE id(n) = {node1.element_id} SET n.embedding = {tripplet_embedding.tolist()}"
            )



def compute_node_embeddings(driver, model):
    with driver.session() as session:
        # Retrieve all nodes
        result = session.run("MATCH (n:Person) RETURN n")
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


def find_most_similar_node(driver, question_embedding):

    with driver.session() as session:
        # Perform the vector search on all nodes based on the question embedding
        result = session.run(
            f"CALL vector_search.search('person_index', 5, {question_embedding.tolist()}) YIELD * RETURN *;"
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


def seed_database(driver):
    with driver.session() as session:
        
        # Clear the database
        session.run("MATCH (n) DETACH DELETE n")
        sleep(1)



        # Create a few nodes
        session.run("CREATE (:Person {name: 'Alice', age: 30})")
        session.run("CREATE (:Person {name: 'Bob', age: 25})")
        session.run("CREATE (:Person {name: 'Charlie', age: 35})")
        session.run("CREATE (:Person {name: 'David', age: 40})")
        session.run("CREATE (:Person {name: 'Eve', age: 20})")
        session.run("CREATE (:Person {name: 'Frank', age: 45})")
        session.run("CREATE (:Person {name: 'Grace', age: 50})")
        session.run("CREATE (:Person {name: 'Hannah', age: 55})")
        session.run("CREATE (:Person {name: 'Jack', age: 65})")
        

        session.run("CREATE (:Person {name: 'Peter', age: 30})")
        session.run("CREATE (:Person {name: 'Peter', age: 60})")
        session.run("CREATE (:Person {name: 'Peter', age: 90})")
        session.run("CREATE (:Person {name: 'John', age: 30})")
        session.run("CREATE (:Person {name: 'John', age: 60})")
        session.run("CREATE (:Person {name: 'John', age: 90})")
        session.run("CREATE (:Person {name: 'Petar', age: 30})")
        session.run("CREATE (:Person {name: 'Petar', age: 60})")
        session.run("CREATE (:Person {name: 'Petar', age: 90})")

        session.run("CREATE (:Bank {name: 'Deutsche Bank AG'})")
        session.run("CREATE (:Bank {name: 'Commerzbank'})")
        session.run("CREATE (:Bank {name: 'Unicredit Bank AG'})")

        session.run("CREATE (:Country {name: 'Germany'})")
        session.run("CREATE (:Country {name: 'Canada'})")

        session.run("CREATE (:City {name: 'Munich'})")


        session.run("MATCH (p:Person {name: 'Peter', age: 30}), (o:Country {name: 'Germany'}) MERGE (p)-[:LIVES_IN]->(o);")
        session.run("MATCH (p:Person {name: 'John', age: 30}), (o:Country {name: 'Germany'}) MERGE (p)-[:LIVES_IN]->(o);")
        session.run("MATCH (p:Person {name: 'Charlie', age: 35}), (o:Country {name: 'Germany'}) MERGE (p)-[:LIVES_IN]->(o);")


        session.run("MATCH (p:Person {name: 'Peter', age: 60}), (b:Bank {name: 'Deutsche Bank AG'}) MERGE (p)-[:IS_CLIENT]->(b);")
        session.run("MATCH (p:Person {name: 'John', age: 60}), (b:Bank {name: 'Commerzbank'}) MERGE (p)-[:IS_CLIENT]->(b);")


        session.run("MATCH (p:Person {name: 'Bob', age: 25}), (o:Country {name: 'Canada'}) MERGE (p)-[:LIVES_IN]->(o);")
        session.run("MATCH (p:Person {name: 'David', age: 40}), (o:Country {name: 'Canada'}) MERGE (p)-[:LIVES_IN]->(o);")
        session.run("MATCH (p:Person {name: 'Eve', age: 20}), (o:Country {name: 'Canada'}) MERGE (p)-[:LIVES_IN]->(o);")

        session.run("MATCH (p:Person {name: 'Frank', age: 45}), (o:City {name: 'Munich'}) MERGE (p)-[:WORKS_IN]->(o);")

  


        session.run("""CREATE VECTOR INDEX person_index ON :Person(embedding) WITH CONFIG {"dimension": 384, "capacity": 1000, "metric": "cos","resize_coefficient": 2}""")



def main(question, tripplets):

    print("The question: ", question)
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))

    # Seed the database with some data
    seed_database(driver)
    
    # Load the SentenceTransformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    if tripplets:
        compute_tripplets_embeddings(driver, model)
    else:
        compute_node_embeddings(driver, model)

    question_embedding = model.encode(question)
    most_similar_node = find_most_similar_node(driver, question_embedding)


if __name__ == "__main__":
    # question = "Is there any Person in the database?"
    # question = "Is there any person Peter?"
    # question = "Is there any person Petar?"
    # question = "Is there any person that lives in Germany?"
    # question = "Is there any Peter that lives in Germany?"
    question = "Is there any Peter that lives in Munich?"

    main(question=question, tripplets=True)
