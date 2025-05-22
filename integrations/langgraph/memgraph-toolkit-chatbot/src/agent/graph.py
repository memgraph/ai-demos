"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from typing import Annotated, TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langchain_memgraph.graphs.memgraph import MemgraphLangChain
from langchain_memgraph import MemgraphToolkit


url = "bolt://localhost:7687"
username = "memgraph"
password = "memgraph"

db = MemgraphLangChain(
    url=url, username=username, password=password, refresh_schema=False
)

llm = init_chat_model("openai:gpt-4.1")
toolkit = MemgraphToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools, name="tools")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
