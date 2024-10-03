import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from prompts.router import router_instructions


def test_router(llm_json_mode: Runnable) -> None:
    test_web_search = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [
            HumanMessage(
                content="Who is favored to win the NFC Championship game in the 2024 season?"
            )
        ]
    )
    test_web_search_2 = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content="What are the models released today for llama3.2?")]
    )
    test_vector_store = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content="What are the types of agent memory?")]
    )
    print(
        json.loads(test_web_search.content),
        json.loads(test_web_search_2.content),
        json.loads(test_vector_store.content),
    )
