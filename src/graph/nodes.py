import json
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_logger
from graph.state import GraphState
from prompts.answer_grader import (answer_grader_instructions,
                                   answer_grader_prompt)
from prompts.grader import doc_grader_instructions, doc_grader_prompt
from prompts.hallucination_grader import (hallucination_grader_instructions,
                                          hallucination_grader_prompt)
from prompts.rag import rag_prompt
from prompts.router import router_instructions
from retrievers.web import web_retriever
from utils.utils import format_docs


def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")

    retriever = web_retriever(state.urls)
    documents: List[Document] = retriever.invoke(state.question)

    return {"documents": documents}


def generate(state: GraphState):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")

    llm = state.llm

    docs_txt = format_docs(state.documents)

    rag_prompt_formated = rag_prompt.format(context=docs_txt, question=state.question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formated)])

    return {"generation": str(generation), "loop_step": state.loop_step + 1}


def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    filtered_docs = []
    web_search = "no"

    for doc in state.documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=doc.page_content, question=state.question
        )
        result = state.llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )

        grade = json.loads(result.content)["binary_score"]

        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT")
            print("DOCUMENT RELEVANT")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            print("DOCUMENT NOT RELEVANT")
            web_search = "yes"
            continue

    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state: GraphState):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    print("---WEB SEARCH---")

    documents = state.documents or []

    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": state.question})
    web_results = "\n".join([doc["content"] for doc in docs])
    documents.append(Document(page_content=web_results))

    return {"documents": documents}


def route_question(state: GraphState):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")

    route_question = state.llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state.question)]
    )

    source = json.loads(route_question.content)["datasource"]

    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")

    if state.web_search == "Yes":
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(state.documents), generation=state.generation
    )

    result = state.llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATE vs QUESTION---")

        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=state.question, generation=state.generation
        )

        result = state.llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )

        grade = json.loads(result.content)["binary_score"]

        print(f"grade: {grade}")

        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state.loop_step <= state.max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "not useful"

    elif state.loop_step <= state.max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"
