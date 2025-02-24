# Agentic GraphRAG

The purpose of the Agentic GraphRAG demo is to be dataset agnostic. Still, you
can use our example dataset to assist you in quick start. 

## How to run

### 1. Prepare the dataset and run Memgraph 

The `setup.sh` script runs Memgraph and imports the dataset from the CYPHERL
file. You can tweak the `setup.sh` script and update the path to point to your
CYPHERL file. 

To run the `setup.sh` script, run the following in the terminal:

```
bash ./setup.sh
```

If you haven't updated the `setup.sh` script, it will load an example AskNews
finance dataset. 

If you prefer not to use `setup.sh` script, make sure you have Memgraph running
with the dataset loaded and `schema-info-enabled` set to `True`.

### 2. Install dependencies

Ensure you have all the required dependencies installed. You can use `pip` to install them. 

```
pip install -r requirements.txt
```

### 3. Set Up environment variables

Create a .env file in the same directory as your script and add your OpenAI API
key:

```
OPENAI_API_KEY=your_openai_api_key
```

or export the OpenAI API key:

```
export OPENAI_API_KEY=your_openai_api_key
```

### 4. Run the script

Use Streamlit to run the script. Open a terminal, navigate to the directory
containing agenticGraphRAG.py, and run:

```
streamlit run agenticGraphRAG.py
```

Besides starting a local web server, the above command will calculate and store
embeddings, communities and community summaries in Memgraph, which will be used
in the retrieval techniques. 

### 5. Access the application

Open your web browser and go to the URL provided in the terminal (usually
http://localhost:8501).

This will launch the Streamlit application, where you can enter your questions
and interact with the Agentic GraphRAG system.

## Example questions

In case you started the app with the provided AskNews finance dataset, here are some questions you can ask:

> [!NOTE]  
> The agentic GraphRAG might not act the same if you run the questions listed about. It runs autonomously and makes decisions along the way, which can lead to a different decision in subsequent runs. 

1. **What can you tell me about this dataset?** -> expected question type: Database -> tools: 1. schema, 2. config

2. **What can you tell me about Coca cola?** -> expected question type: Retrieval -> tools: 1. Cypher, 2. Vector relevance expansion

3. **How is coca cola connected in the graph, what is relationship with other companies?** -> expected question type: Structure -> tools: 1. Cypher, 2. Vector relevance expansion

4. **What level of logging does Memgraph have enabled?** -> expected question type: Database -> tools: 1. config, 2. schema

5. **Is Coca Cola in the most important nodes?** -> expected question type: Global -> tools: 1. PageRank, 2. Community -> LLM picks the 10 most important nodes and Coca Cola is not among them; **Is Coca Cola in the 1000 most important nodes?** -> Yes

