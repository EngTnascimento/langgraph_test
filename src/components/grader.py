import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import Runnable

from prompts.grader import doc_grader_instructions, doc_grader_prompt


def grade(question: str, retriever: RetrieverLike, llm_json_mode: Runnable):
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    doc_grader_prompt_formatted = doc_grader_prompt.format(
        document=doc_txt, question=question
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    return result
