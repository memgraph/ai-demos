# Vector Search

## How to run

Follow these steps to run the `vector_search.py` script:

### 1. Install dependencies

Ensure you have all the required dependencies installed. You can use `pip` to install them. 

```
pip install -r requirements.txt
```

### 2. Set Up environment variables

Create a .env file in the same directory as your script and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
```

### 4. Configure Memgraph connection

Depending on where your Memgraph is running, adjust the client IP and port 

### 5. Run the script

Open a terminal, navigate to the directory containing `vector_search.py`, and run:

```
python vector_search.py
```