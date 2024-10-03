# LangGraph Adaptive RAG with Local Embeddings and Web Search

This repository demonstrates how to implement a **Retrieval-Augmented Generation (RAG)** system using **LangGraph**, incorporating both local embeddings and web search as part of the retrieval process.

The tutorial is based on the official LangGraph documentation [here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#web-search-tool).

## Features

- **RAG Pipeline**: Combines retrieval of documents with a language model for answering queries.
- **Local Embeddings**: Supports generating and querying local embeddings.
- **Web Search**: Integrates real-time web search for enhanced context retrieval.
- **Flexible Execution**: Dynamic switching between local and web-based retrieval based on query type.

## Prerequisites

- **Python 3.8+**
- **LangGraph**: Install from the official LangGraph GitHub or via pip.
- **Embeddings Model**: For local embeddings (e.g., SentenceTransformers).
- **Web Search API**: Optionally set up a search engine API key (Google, Bing, etc.) if you want to enable web search.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/langgraph-adaptive-rag.git
    cd langgraph-adaptive-rag
    ```

2. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment variables:
    - `SEARCH_API_KEY`: Your web search API key (if applicable).
    - `MODEL_PATH`: Path to your local embedding model.
  
   Example `.env` file:
    ```bash
    SEARCH_API_KEY=your_web_search_api_key
    MODEL_PATH=/path/to/local/embedding/model
    ```

## Usage

To run the Adaptive RAG system, follow these steps:

1. **Run the RAG pipeline**:
    ```bash
    python run_rag.py --query "What are the recent advancements in AI?"
    ```

    This will:
    - Use local embeddings for document retrieval.
    - Use web search for additional context if needed.
    - Generate an answer based on both sources.

2. **Configure Local Embeddings**:
    You can customize the local embedding model used by editing `config.yaml`.

    ```yaml
    embeddings_model:
      name: sentence-transformers/all-MiniLM-L6-v2
    ```

## Project Structure

```
├── data/                # Local documents for embedding retrieval
├── langgraph/           # LangGraph pipeline code
├── config.yaml          # Configuration for local embeddings and search
├── .env                 # Environment variables for web search and model paths
├── run_rag.py           # Main script to execute RAG
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Customization

You can adjust the RAG pipeline to:
- Use a different embeddings model.
- Modify the search tool (Google, Bing, or others).
- Change how LangChain handles the query flow between local and web retrieval.

For more details, refer to the LangGraph [documentation](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#web-search-tool).

