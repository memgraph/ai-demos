# :sparkles: Memgraph AI Demos :sparkles:

There are various ways how AI can be utilized to interact with a graph database
such as Memgraph. Whether you're creating a knowledge graph from unstructured
data, figuring out the best retrieval strategy, implementing GraphRAG or
creating a fully autonomous agent - the possiblities are endless.

This repository is a collection of our demos and examples and we'll continue
growing it as we learn. Here's what we have so far:

## [Knowledge graph creation](./knowledge-graph-creation/)

This directory contains demos focused on building knowledge
graphs from unstructured data. These examples illustrate how to extract entities
and relationships to form structured graphs, enhancing data comprehension and
accessibility.

**Contents:**
- **:bulb: Demo: [Catcher in the Rye](./knowledge-graph-creation/catcher-in-the-rye/)**
  - This demo focuses on constructing a knowledge graph from the
    text of "Catcher in the Rye". It demonstrates the extraction of entities and
    relationships from the summary of the book, showcasing how to model and
    build context from the data within Memgraph.
  - **:mag_right: Key Features:**
    - Text preprocessing and entity recognition using [SpaCy](https://spacy.io/)
    - Building relationships from character interactions and plot developments
    - Visualization of the resulting knowledge graph to explore the story's
      dynamics using [Memgraph Lab](https://memgraph.com/docs/data-visualization).

- **:bulb: Demo: [Game of Thrones](./knowledge-graph-creation/game-of-thrones/)**
  - This demo illustrates the process of building a knowledge
    graph from "Game of Thrones" data. It involves extracting key entities such
    as characters, houses, and locations, and mapping their complex
    relationships to provide insights into the intricate world of the series.
  - **:mag_right: Key Features:**
    - Extraction of entities like characters, houses, and locations
    - Mapping of relationships including alliances, rivalries, and family ties
    - Visualization of the complex network within the "Game of Thrones" universe using [Memgraph Lab](https://memgraph.com/docs/data-visualization)

**:book: Additional resources**
- [More about knowledge graphs in Memgraph](https://memgraph.com/docs/data-modeling/knowledge-graph)
- [Blog post "How to Extract Entities and Build a Knowledge Graph with Memgraph and SpaCy"](https://memgraph.com/blog/extract-entities-build-knowledge-graph-memgraph-spacy)
- [Youtube video "Knowledge Graph Creation by Entity Extraction in Memgraph"](https://www.youtube.com/watch?v=HYYhtKC2jyA)


## [Retrieval](./retrieval/)

This directory contains demos focused on various retrieval strategies to efficiently query and extract relevant information from a knowledge graph. These examples illustrate how to leverage Memgraph's capabilities to perform advanced searches and retrieve data based on specific criteria.

**Contents:**
- **:bulb: Demo: [Vector Search](./retrieval/vector-search/)**
  - This demo showcases the use of vector search in Memgraph to find semantically similar nodes based on embeddings. It highlights the process of encoding node properties and performing similarity searches to retrieve relevant data.
  - **:mag_right: Key Features:**
    - Encoding node properties into embeddings
    - Performing vector searches to find similar nodes
    - Advanced querying capabilities to explore the retrieved data

## [GraphRAG](./graph-rag/) 

This directory contains demos focused on building a Graph-based Retrieval-Augmented Generation (GraphRAG) system that uses Memgraph, to perform knowledge graph-based question answering. The demo illustrates how to build an end-to-end GraphRAG system using Memgraph.

**Contents:**
- **:bulb: Demo: [GraphRAG](./graph-rag/graphRAG.ipynb)**
  - This demo demonstrates the implementation of a GraphRAG system using a Game of Thrones dataset. It involves enriching the knowledge graph with unstructured data, performing vector searches, and using LLMs to answer questions based on the graph data.
  - **:mag_right: Key Features:**
    - Enriching the knowledge graph with unstructured data
    - Performing vector searches to find relevant nodes
    - Using LLMs to answer questions based on the graph data
    - Embedding the node properties and labels
    - Performing the relevance expansions with Memgraph BFS algorithm

## [Agentic GraphRAG](./agentic-graph-rag/)

This directory contains demos focused on building an autonomous agent using the GraphRAG system. These examples illustrate how to create an agent that can interact with a knowledge graph, retrieve relevant information, and generate responses based on the data. The agents are dataset agnostic. 

**Contents:**
- **:bulb: Demo: [Agentic GraphRAG](./agentic-graph-rag/agenticGraphRAG.py)**
  - This demo showcases the creation of an autonomous agent using the GraphRAG system. It highlights the process of integrating Memgraph, Sentence Transformers, and OpenAI's GPT models to build an agent that can answer questions and perform tasks based on the knowledge graph.
  - **:mag_right: Key Features:**
    - Building an autonomous agent using GraphRAG
    - Integrating Memgraph, Sentence Transformers, and OpenAI's GPT models
    - Advanced querying and response generation based on the knowledge graph

## [Integrations](./integrations/)

This directory contains integrations that demonstrate how to
connect and utilize third-party frameworks with Memgraph. These examples
highlight the process of leveraging tools like LlamaIndex and LangChain to
process unstructured data, extract entities and relationships and build
knowledge graphs seamlessly within Memgraph.

**Contents:**
- **:bulb: Demo: [Langchain](./integrations/langchain/)**
  - This demo showcases the integration of LangChain with Memgraph
    to create a knowledge graph from unstructured data. It highlights the use of
    LangChain's framework to process text, extract entities and relationships,
    and store them within Memgraph for advanced querying and analysis.
  - **:mag_right: Key Features:**
    - Utilization of LangChain for text processing and entity extraction
    - Construction of a knowledge graph within Memgraph
    - Advanced querying capabilities to explore the structured data

- **:bulb: Demo: [LlamaIndex](./integrations/llamaindex/)**
  - This demo demonstrates the use of LlamaIndex with Memgraph to
    build a knowledge graph from unstructured data. It showcases the framework's
    ability to parse complex documents, extract meaningful entities and
    relationships and represent them as a knowledge graph in Memgraph.
  - **:mag_right: Key Features:**
    - Parsing of complex documents to extract entities and relationships
    - Integration with Memgraph to construct and store the knowledge graph
    - Visualization and querying of the graph to derive insights from the data
      using [Memgraph Lab](https://memgraph.com/docs/data-visualization)

**:book: Additional resources**
- [LangChain & Memgraph](https://memgraph.com/docs/ai-ecosystem/graph-rag#langchain)
- ["Improved Knowledge Graph Creation with LangChain and LlamaIndex"](https://memgraph.com/blog/improved-knowledge-graph-creation-langchain-llamaindex)
- [LlamaIndex & Memgraph](https://memgraph.com/docs/ai-ecosystem/graph-rag#llamaindex)
- ["How to build GenAI apps with LlamaIndex and Memgraph"](https://memgraph.com/webinars/how-to-build-genai-apps-with-llamaindex-and-memgraph)