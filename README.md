# LangGraph Adaptive RAG with Local Embeddings and Web Search

This project implements an **Adaptive Retrieval-Augmented Generation (RAG)** system using **LangGraph**. The system enhances traditional question-answering by combining two types of retrieval methods:

1. **Local Embeddings Retrieval**: Using pre-built embeddings from local documents, it retrieves relevant information to answer user queries.
2. **Web Search**: For queries that require additional or up-to-date information, the system integrates a web search tool to fetch real-time data from the web.

The adaptive nature of this system allows it to dynamically decide whether to rely on local embeddings or perform web searches, based on the nature of the query.

This approach is useful in scenarios where both static and dynamic information are important for generating the best possible response. By merging local and web-based retrieval, it provides a more comprehensive and context-aware answer to complex questions.

For more details on the underlying techniques, refer to the official LangGraph documentation on [Adaptive RAG with Local Embeddings and Web Search](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/#web-search-tool).

