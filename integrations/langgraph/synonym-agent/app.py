from workflows import run_workflow
import streamlit as st
import neo4j
from streamlit_agraph import agraph, Node, Edge, Config


# Function to render a graph using streamlit-agraph from Neo4j query results
def render_graph(query_result):
    nodes = []
    edges = []

    # Extract nodes and relationships from query result
    for record in query_result:
        for key, value in record.items():
            if isinstance(value, neo4j.graph.Node):
                nodes.append(
                    Node(
                        id=str(value["id"]),
                        label=value["name"],
                        size=25,
                        selectable=True,
                    )
                )
            elif isinstance(value, neo4j.graph.Relationship):
                edges.append(
                    Edge(
                        source=str(value["start"]),
                        target=str(value["end"]),
                        label=value.get("type", ""),
                    )
                )

    # Only render if there are nodes or edges
    if nodes or edges:
        config = Config(
            width=800,
            height=600,
            directed=True,
            physics=True,
            hierarchical=False,
            selectable=True,
            interaction={"dragNodes": True, "dragView": True, "zoomView": True},
        )
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("No graph data found in query result.")


def generate_frontend():
    # Streamlit App Title
    st.title("Memgraph LangGraph Chatbot")

    # Text Input for User Question
    user_question = st.text_input("Enter your question:")

    # Checkboxes for display options
    plain_answer = st.checkbox("Plain Answer", value=True)
    query_answer = st.checkbox("Query Answer", value=True)
    graph_view = st.checkbox("Graph View", value=True)

    # Run Workflow on Button Click
    if st.button("Generate Answer"):
        if user_question:
            result = run_workflow(user_question)

            # Displaying Question
            st.markdown(f"**Question:** {result['question']}")

            # Displaying based on checkbox selection
            if plain_answer:
                st.markdown(f"**Final Answer:** {result['final_answer']}")
            if query_answer:
                st.markdown(
                    f"**Query Result:**\n```json\n{result['query_result']}\n```"
                )
            if graph_view:
                render_graph(result["query_result"])

            # Always show the generated query
            st.markdown(
                f"**Generated Cypher Query:**\n```cypher\n{result['cypher_query']}\n```"
            )
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    generate_frontend()
