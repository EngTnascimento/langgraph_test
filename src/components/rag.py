from langchain_core.messages import HumanMessage
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable

from prompts.rag import rag_prompt
from utils.utils import format_docs


def rag(question: str, llm: Runnable, retriever: RetrieverLike):
    docs = retriever.invoke(question)
    docs_txt = format_docs(docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    print(generation.content)
    return generation, docs_txt
