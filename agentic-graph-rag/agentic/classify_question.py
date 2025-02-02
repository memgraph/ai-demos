from typing import Dict

# - Path
# - Path: The query seeks information about the path between two entities, such as shortest path or existence of a path.
# - Path: Is there a any path between John and Mary? 
def get_classify_question_prompt(config: Dict[str, bool]):
    """Dynamically adjusts the CLASSIFY_QUESTION_PROMPT based on graph-algorithms-enabled flag."""

    # Base prompt
    base_prompt = """
    Classify the following user question into query type

    Query Types:
    - Retrieval
    - Database
    - Other
    """

    # Include "Structure" only if graph-algorithms-enabled is present
    if config.get("vector-search-enabled", False):
        base_prompt += """
        - Structure
        """

    # Include "Global" only if graph-algorithms-enabled is present
    if config.get("graph-algorithms-enabled", False):
        base_prompt += """
        - Global
        """

    base_prompt += """

    Each type of question has different characteristics.
    - Retrieval: Direct Lookups, specific and well-defined. The query seeks information about a single entity (node or relationship). 
    """

    # Add "Structure" and "Global" descriptions only if enabled
    if config.get("vector-search-enabled", False):
        base_prompt += """
        - Structure: Exploratory, the query seeks information about the structure of the graph, close relationships between entities, or properties of nodes.
        """

    if config.get("graph-algorithms-enabled", False):
        base_prompt += """
        - Global: The query seeks context about the entire graph, community, such as the most important node or global trends in graph. 
        """

    base_prompt += """
    - Database: The query seeks statistical information about the database, such as index information, node count, or relationship count, config etc.
    - Other: If the question does not fit into any of the above categories, if it is ambiguous or unclear, joke or irrelevant.

    Example of a questions for each type:
    - Retrieval: How old is a person with the name "John"? 
    """

    # Include "Structure" and "Global" examples only if enabled
    if config.get("vector-search-enabled", False):
        base_prompt += """
        - Structure: Does John have a job? Is John a friend of Mary?
        """

    if config.get("graph-algorithms-enabled", False):
        base_prompt += """
        - Global: What is the most important node in the graph?
        """

    base_prompt += """
    - Database: What indexes does Memgraph have?
    - Other: What is the meaning of life?

    In the explanation, provide a brief description of the type of question, and why you classified it as such. 
    
    If in the question it is explicitly said which query type to use, then pick that one instead.

    The question is in <Question> </Question> format.
    """

    return base_prompt
