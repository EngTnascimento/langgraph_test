import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from prompts.answer_grader import (answer_grader_instructions,
                                   answer_grader_prompt)


def grade_answer(question: str, answer: str, llm_json_mode: Runnable):
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        question=question, generation=answer
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )
    return result
