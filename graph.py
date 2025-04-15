from typing import Annotated

from typing_extensions import TypedDict
import llm_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from models import LlmResponse, Pii

import re

from langgraph.checkpoint.memory import MemorySaver

llm = llm_model.llm

class DetectorResponse(BaseModel):
    """Response of PII detector"""
    res: Annotated[list[str], ..., "PII"]

class State(TypedDict):
    content: str
    names: list[str]
    phones: list[str]
    addresses: list[str]
    emails: list[str]
    rrns: list[str]

def name_detector(state: State):
    prompt = open("prompts/name_detector.txt", encoding="utf-8").read()
    structured_llm = llm.with_structured_output(DetectorResponse)
    return {"names": structured_llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["content"]}
    ]).res}

def phone_detector(state: State):
    prompt = open("prompts/phone_detector.txt", encoding="utf-8").read()
    structured_llm = llm.with_structured_output(DetectorResponse)
    return {"phones": structured_llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["content"]}
    ]).res}

def address_detector(state: State):
    prompt = open("prompts/address_detector.txt", encoding="utf-8").read()
    structured_llm = llm.with_structured_output(DetectorResponse)
    return {"addresses": structured_llm.invoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["content"]}
    ]).res}
    
def email_detector(state: State):
    content = state["content"]
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content)
    return {"emails": emails}

def rrn_detector(state: State):
    content = state["content"]
    rrns = re.findall(r"\d{2}(?:[0]\d|[1][0-2])(?:[0][1-9]|[1-2]\d|[3][0-1])[-]*[1-4]\d{6}", content)
    return {"rrns": rrns}


def merger(state: State):
    return {
        "names": state["names"], 
        "phones": state["phones"], 
        "addresses": state["addresses"],
        "emails": state["emails"],
        "rrns": state["rrns"],
    }

builder = StateGraph(State)

builder.add_node("name_detector", name_detector)
builder.add_node("phone_detector", phone_detector)
builder.add_node("address_detector", address_detector)
builder.add_node("email_detector", email_detector)
builder.add_node("rrn_detector", rrn_detector)
builder.add_node("merger", merger)

builder.add_edge(START, "name_detector")
builder.add_edge(START, "phone_detector")
builder.add_edge(START, "address_detector")
builder.add_edge(START, "email_detector")
builder.add_edge(START, "rrn_detector")
builder.add_edge(["name_detector", "phone_detector", "address_detector", "email_detector", "rrn_detector"], "merger")
builder.add_edge("merger", END)
builder.compile()

graph = builder.compile(checkpointer=MemorySaver())

async def detect_pii(content: str) -> LlmResponse:
    state = State(content=content, names=[], phones=[], addresses=[])
    config = {"configurable": {"thread_id": "test"}}
    async for event in graph.astream(state, config=config):
        print(event)
    res_state = graph.get_state(config=config)
    print(res_state)
    return LlmResponse(
        pii=Pii(
            name=res_state.values["names"],
            phone=res_state.values["phones"],
            address=res_state.values["addresses"],
            email=res_state.values["emails"],
            rrn=res_state.values["rrns"],
        ),
        prompt_tokens=0,
        completion_tokens=0,
        calls=0,
    )

if __name__ == "__main__":
    # content = open("test.txt", encoding="utf-8").read()
    # import asyncio
    # print(asyncio.run(detect_pii(content)))
    
    print(re.findall(r"\d{2}(?:[0]\d|[1][0-2])(?:[0][1-9]|[1-2]\d|[3][0-1])[-]*[1-4]\d{6}", "980113-1234567 010101-2312345"))