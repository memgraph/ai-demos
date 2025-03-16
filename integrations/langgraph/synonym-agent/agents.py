import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import MemgraphGraph
from langchain_community.chains.graph_qa.prompts import (
    MEMGRAPH_GENERATION_PROMPT,
)
import yaml


URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
USER = os.getenv("MEMGRAPH_USER", "")
PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")
OPENAI_API_KEY = "<YOUR OPENAI API KEY>"


# Initialize MemgraphGraph
graph = MemgraphGraph(
    url=URI,
    username=USER,
    password=PASSWORD,
)


class BusinessSynonymRule:
    def __init__(self, label: str, prop: str, explanation: str):
        self.label = label
        self.prop = prop
        self.explanation = explanation

    def __repr__(self):
        return f"For label :{self.label}, property {self.prop} -> {self.explanation}"


def load_business_rules(yaml_file="business_rules.yaml"):
    """
    Loads business synonym rules from a YAML file.

    :param yaml_file: Path to the YAML file containing business rules.
    :return: A list of BusinessSynonymRule objects.
    """
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Error: The file '{yaml_file}' does not exist.")

    try:
        with open(yaml_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            rules_data = data.get("configuration", {}).get("business_rules", [])

            # Convert YAML data to BusinessSynonymRule objects
            business_rules = [
                BusinessSynonymRule(rule["label"], rule["prop"], rule["explanation"])
                for rule in rules_data
            ]
            return business_rules

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{yaml_file}': {e}")
    except KeyError as e:
        raise ValueError(f"Missing key in YAML file: {e}")


# Load the business rules
business_rules = load_business_rules()


def clean_cypher_query(cypher_query: str) -> str:
    """Cleans LLM-generated Cypher query by removing markdown code block markers."""
    return cypher_query.replace("```", "").replace("cypher\n", "")


def initialize_graph_context(state):
    """Agent provides the state with the schema."""
    return {"schema": graph.get_schema}


def generate_cypher_query(state):
    """Agent for generating Cypher queries."""
    question = state["question"]
    schema = state["schema"]

    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    cypher_query = llm.invoke(
        MEMGRAPH_GENERATION_PROMPT.format(schema=schema, question=question)
    )
    cleaned_cypher_query = clean_cypher_query(cypher_query.content)

    return {"cypher_query": cleaned_cypher_query, "initial_query": cleaned_cypher_query}


def business_synonym_reasoning(state):
    """Rewrites the Cypher query based on predefined business synonym rules."""
    cypher_query = state["cypher_query"]
    schema = state["schema"]
    rules = "\n".join([str(x) for x in business_rules])

    BUSINESS_SYNONYM_PROMPT = """Your task is to analyze and, if necessary, rewrite 
    the given Cypher query based on the following business synonym rules. 
    If none of the rules apply, return the query unchanged.
    
    Given the following schema of the graph:
    {schema}
    
    The query needs to be rewritten if any of the following rules apply:
    {rules}

    Given the following Cypher query:
    {cypher_query}

    Return the rewritten query (if modified) or the original query if no changes were needed.
    Return no additional parts except just the cypher query, no apologies, additional text, ommit everything.
    """

    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)

    # Inject business rules into the prompt (to be defined separately)
    formatted_prompt = BUSINESS_SYNONYM_PROMPT.format(
        schema=schema,
        rules=rules,
        cypher_query=cypher_query,
    )

    revised_query = llm.invoke(formatted_prompt)
    cleaned_cypher_query = clean_cypher_query(revised_query.content)

    return {"cypher_query": cleaned_cypher_query}


def execute_cypher_query(state):
    """Executes the Cypher query on Memgraph."""
    cypher_query = state["cypher_query"]
    try:
        # result = graph.query(cypher_query)
        data, _, _ = graph._driver.execute_query(
            cypher_query,
            database_=graph._database,
            parameters_={},
        )
        # json_data = [r.data() for r in data]
        return {"query_result": data}
    except Exception as e:
        return {"query_result": f"Error: {str(e)}"}


def generate_human_readable_response(state):
    """Generates a human-readable response from the query result."""
    question = state["question"]
    context = state["query_result"]

    MEMGRAPH_QA_TEMPLATE = """Your task is to form nice and human
    understandable answers. The information part contains the provided
    information that you must use to construct an answer.
    The provided information is authoritative, you must never doubt it or try to
    use your internal knowledge to correct it. Make the answer sound as a
    response to the question. Do not mention that you based the result on the
    given information. Here is an example:

    Question: Which managers own Neo4j stocks?
    Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
    Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

    Follow this example when generating answers. If the provided information is
    empty, say that you don't know the answer. If anything is in the context, meaning
    that Memgraph returned some of the results, please try to generate a meaningful answer.
    If there is really nothing in the context, then you can say that you don't know 
    the answer.

    Information:
    {context}

    Question: {question}
    Helpful Answer:"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"], template=MEMGRAPH_QA_TEMPLATE
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    final_answer = llm.invoke(qa_prompt.format(context=context, question=question))
    return {"final_answer": final_answer.content}
