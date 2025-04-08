from colorama import Fore, Style
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph
from agents import (
    business_synonym_reasoning,
    execute_cypher_query,
    generate_cypher_query,
    generate_human_readable_response,
    initialize_graph_context,
)
from typing import TypedDict


class WorkflowState(TypedDict):
    question: str
    schema: str
    initial_query: str
    business_reasoning: str
    cypher_query: str
    query_result: str
    final_answer: str


def get_business_reasoning_workflow():
    # Define the state graph
    graph_workflow = StateGraph(WorkflowState)
    graph_workflow.add_node(
        "initialize_graph_context", RunnableLambda(initialize_graph_context)
    )
    graph_workflow.add_node("generate_cypher", RunnableLambda(generate_cypher_query))
    graph_workflow.add_node(
        "synonym_reasoning", RunnableLambda(business_synonym_reasoning)
    )
    graph_workflow.add_node("execute_cypher", RunnableLambda(execute_cypher_query))
    graph_workflow.add_node(
        "generate_response", RunnableLambda(generate_human_readable_response)
    )

    graph_workflow.add_edge("initialize_graph_context", "generate_cypher")
    graph_workflow.add_edge("generate_cypher", "synonym_reasoning")
    graph_workflow.add_edge("synonym_reasoning", "execute_cypher")
    graph_workflow.add_edge("execute_cypher", "generate_response")

    graph_workflow.set_entry_point("initialize_graph_context")
    workflow = graph_workflow.compile()

    return workflow


def get_basic_cypher_workflow():
    # Define the state graph
    graph_workflow = StateGraph(WorkflowState)
    graph_workflow.add_node(
        "initialize_graph_context", RunnableLambda(initialize_graph_context)
    )
    graph_workflow.add_node("generate_cypher", RunnableLambda(generate_cypher_query))
    graph_workflow.add_node("execute_cypher", RunnableLambda(execute_cypher_query))
    graph_workflow.add_node(
        "generate_response", RunnableLambda(generate_human_readable_response)
    )

    graph_workflow.add_edge("initialize_graph_context", "generate_cypher")
    graph_workflow.add_edge("generate_cypher", "execute_cypher")
    graph_workflow.add_edge("execute_cypher", "generate_response")

    graph_workflow.set_entry_point("initialize_graph_context")
    workflow = graph_workflow.compile()

    return workflow


def pick_tool(user_question):
    return get_business_reasoning_workflow()


def run_workflow(user_question):
    initial_state = {"question": user_question}
    workflow = pick_tool(user_question)
    final_output = workflow.invoke(initial_state)

    print(
        f"{Fore.YELLOW}Question:{Style.RESET_ALL} {final_output['question']}\n"
        f"{Fore.BLUE}Initial query:{Style.RESET_ALL} {final_output['initial_query']}\n"
        f"{Fore.BLUE}Query:{Style.RESET_ALL} {final_output['cypher_query']}\n"
        f"{Fore.GREEN}Answer:{Style.RESET_ALL} {final_output['final_answer']}\n"
    )
    print()

    return final_output


if __name__ == "__main__":
    run_workflow("How many nodes are there in the graph?")
