import operator
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from pydantic import BaseModel


class GraphState(BaseModel):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    model_config = {"arbitrary_types_allowed": True}
    llm_json_mode: Runnable  # model in json mode
    llm: Runnable  # model to use
    question: str  # User question
    urls: List[str] = []  # websites to retrieve
    generation: str = ""  # LLM generation
    web_search: str = "no"  # Binary decision to run web search
    max_retries: int = 3  # Max number of retries for answer generation
    answers: int = 0  # Number of answers generated
    loop_step: Annotated[int, operator.add] = 0
    documents: List[Document] = []  # List of retrieved documents
