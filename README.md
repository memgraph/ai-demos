# :sparkles: Memgraph AI Demos :sparkles:

There are various ways how AI can be utilized to interact with a graph database
such as Memgraph. Whether you're creating a knowledge graph from unstructured
data, figuring out the best retrieval strategy, implementing GraphRAG or
creating a fully autonomous agent - the possiblities are endless.

This repository is a collection of our demos and examples and we'll continue
growing it as we learn. Here's what we have so far:

## [Knowledge graph creation](./knowledge-graph-creation/)

**Description:** This directory contains demos focused on building knowledge
graphs from unstructured data. These examples illustrate how to extract entities
and relationships to form structured graphs, enhancing data comprehension and
accessibility.

**Contents:**
- **Demo: [Catcher in the Rye](./knowledge-graph-creation/catcher-in-the-rye/)**
  - **Overview:** This demo focuses on constructing a knowledge graph from the
    text of "Catcher in the Rye." It demonstrates the extraction of entities and
    relationships from the summary of the book, showcasing how to model and
    build context from the data within Memgraph.
  - **Key Features:**
    - Text preprocessing and entity recognition using [SpaCy](https://spacy.io/)
    - Building relationships from character interactions and plot developments
    - Visualization of the resulting knowledge graph to explore the story's
      dynamics using [Memgraph Lab](https://memgraph.com/docs/data-visualization).

- **Demo: [Game of Thrones](./knowledge-graph-creation/game-of-thrones/)**
  - **Overview:** This demo illustrates the process of building a knowledge
    graph from "Game of Thrones" data. It involves extracting key entities such
    as characters, houses, and locations, and mapping their complex
    relationships to provide insights into the intricate world of the series.
  - **Key Features:**
    - Extraction of entities like characters, houses, and locations
    - Mapping of relationships including alliances, rivalries, and family ties
    - Visualization of the complex network within the "Game of Thrones" universe using [Memgraph Lab](https://memgraph.com/docs/data-visualization)

**Additional resources**
- [More about knowledge graphs in Memgraph](https://memgraph.com/docs/data-modeling/knowledge-graph)
- [Blog post "How to Extract Entities and Build a Knowledge Graph with Memgraph and SpaCy"](https://memgraph.com/blog/extract-entities-build-knowledge-graph-memgraph-spacy)
- [Youtube video "Knowledge Graph Creation by Entity Extraction in Memgraph"](https://www.youtube.com/watch?v=HYYhtKC2jyA)


## [Retrieval](./retrieval/)

#TODO: write down/link to docs why this is important and list examples we have. 

## [GraphRAG](./graph-rag/) 

[GraphRAG](https://memgraph.com/docs/ai-ecosystem/graph-rag) demo provides an implementation of RAG system

## [Agentic GraphRAG](./agentic-graph-rag/)

#TODO: write down/link to docs why this is important and list examples we have. 

## [Integrations](./integrations/)

**Description:** This directory contains integrations that demonstrate how to
connect and utilize third-party frameworks with Memgraph. These examples
highlight the process of leveraging tools like LlamaIndex and LangChain to
process unstructured data, extract entities and relationships and build
knowledge graphs seamlessly within Memgraph.

**Contents:**
- **Demo: [Langchain](./integrations/langchain/)**
  - **Overview:** This demo showcases the integration of LangChain with Memgraph
    to create a knowledge graph from unstructured data. It highlights the use of
    LangChain's framework to process text, extract entities and relationships,
    and store them within Memgraph for advanced querying and analysis.
  - **Key Features:**
    - Utilization of LangChain for text processing and entity extraction
    - Construction of a knowledge graph within Memgraph
    - Advanced querying capabilities to explore the structured data

- **Demo: [LlamaIndex](./integrations/llamaindex/)**
  - **Overview:** This demo demonstrates the use of LlamaIndex with Memgraph to
    build a knowledge graph from unstructured data. It showcases the framework's
    ability to parse complex documents, extract meaningful entities and
    relationships and represent them as a knowledge graph in Memgraph.
  - **Key Features:**
    - Parsing of complex documents to extract entities and relationships
    - Integration with Memgraph to construct and store the knowledge graph
    - Visualization and querying of the graph to derive insights from the data
      using [Memgraph Lab](https://memgraph.com/docs/data-visualization)

**Additional resources**
- [LangChain & Memgraph](https://memgraph.com/docs/ai-ecosystem/graph-rag#langchain)
- ["Improved Knowledge Graph Creation with LangChain and LlamaIndex"](https://memgraph.com/blog/improved-knowledge-graph-creation-langchain-llamaindex)
- [LlamaIndex & Memgraph](https://memgraph.com/docs/ai-ecosystem/graph-rag#llamaindex)
- ["How to build GenAI apps with LlamaIndex and Memgraph"](https://memgraph.com/webinars/how-to-build-genai-apps-with-llamaindex-and-memgraph)