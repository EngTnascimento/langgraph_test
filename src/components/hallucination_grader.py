import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from prompts.hallucination_grader import (hallucination_grader_instructions,
                                          hallucination_grader_prompt)


def grade_hallucination(llm_json_mode: Runnable, docs_txt, generation):
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    result_json = json.loads(result.content)
    return result_json
