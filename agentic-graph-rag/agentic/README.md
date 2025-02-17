# Agentic GraphRAG

## How to run

Follow these steps to run the `agenticGraphRAG.py` script:

### 1. Prepare the dataset and run Memgraph 

If you are not running Memgraph and do not have a dataset, change to path in the `setup.sh` script to point to `cypherl` dataset file and run the script, this will load the dataset and start Memgraph. 

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

### 4. Run the script

Use Streamlit to run the script. Open a terminal, navigate to the directory
containing agenticGraphRAG.py, and run:

```
streamlit run agenticGraphRAG.py
```


### 5. Access the application

After running the above command, Streamlit will start a local web server. Open
your web browser and go to the URL provided in the terminal (usually
http://localhost:8501).

This will launch the Streamlit application, where you can enter your questions
and interact with the Agentic GraphRAG system.