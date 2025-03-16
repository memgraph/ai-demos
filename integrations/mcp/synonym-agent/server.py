import ollama
from mcp.server.fastmcp import FastMCP

# NOTE: Add/export {{path_to_project}}/integrations/langgraph/synonym-agent/ to
# PYTHONPATH.
from workflows import run_workflow


def prompt_llama3(content):
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )
    return response["message"]["content"]


mcp = FastMCP("Synonym Agent")


# Simple example of a prompt under MCP.
@mcp.prompt()
def prompt_llama3_via_mcp(question: str) -> str:
    return prompt_llama3(question)


# NOTE: At the time of implementation (2025-03-16), MCP does NOT support
# complex workflows or tree of thoughts -> all reasoning logic has to be
# implemented under a single prompt. Using langgraph workflow example.
@mcp.prompt()
def answer_business_specific_question(question: str) -> str:
    return run_workflow(question)["final_answer"]
