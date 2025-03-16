import ollama
from mcp.server.fastmcp import FastMCP


def prompt(content):
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

# NOTE: At the time of implementation (2025-03-16), MCP does NOT supprot
# complex workflows or tree of thought -> all resoning logic has to be
# implemented under a single prompt.
@mcp.prompt()
def answer_business_specific_question(question: str) -> str:
    return prompt(question)
