import os

from IPython.display import Image, display
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from graph.nodes import (decide_to_generate, generate, grade_documents,
                         grade_generation_v_documents_and_question, retrieve,
                         route_question, web_search)
from graph.state import GraphState
from retrievers.web import web_retriever
from utils.utils import _set_env

_set_env("TAVILY_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

retriever = web_retriever(urls)

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)


workflow.set_conditional_entry_point(
    route_question, {"websearch": "websearch", "vectorstore": "retrieve"}
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"websearch": "websearch", "generate": "generate"},
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

graph = workflow.compile()

png_data = graph.get_graph().draw_mermaid_png()
with open("mermaid_graph.png", "wb") as file:
    file.write(png_data)

display(Image(png_data))

local_llm = "llama3.2:3b-instruct-fp16"
llm: Runnable = ChatOllama(model=local_llm, temperature=0)
llm_json_mode: Runnable = ChatOllama(model=local_llm, temperature=0, format="json")

inputs = {
    "max_retries": 3,
    "question": "What are the types of agent memory?",
    "llm": llm,
    "llm_json_mode": llm_json_mode,
    "urls": urls,
}

for event in graph.stream(inputs, stream_mode="values"):
    print(event)
