import asyncio
import textwrap
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

import llm_model
from models import Pii, LlmResponse

CHUNK_SIZE = 1000

llm = llm_model.llm

system_template = SystemMessagePromptTemplate.from_template(
    "당신은 한국 개인정보 식별 전문가입니다. 정확한 개인정보 형식만 인식하세요. 형식이 일치하지 않으면 절대 포함하지 마세요. 틀리면 작업 실패로 간주됩니다."
)
human_template = HumanMessagePromptTemplate.from_template(
    open("prompts/pii_detector.txt", encoding="utf-8").read()
)

prompt = ChatPromptTemplate.from_messages([system_template, human_template])
output_parser = PydanticOutputParser(pydantic_object=Pii)

chain = prompt | llm | output_parser


def split_text(text: str) -> list[str]:
    return textwrap.wrap(text, CHUNK_SIZE, break_long_words=False, replace_whitespace=False)


def merge_results(results: list[Pii]) -> Pii:
    def unique(flat_list):
        return list(set(flat_list))

    return Pii(
        name=unique([item for res in results for item in res.name]),
        phone=unique([item for res in results for item in res.phone]),
        rrn=unique([item for res in results for item in res.rrn]),
        email=unique([item for res in results for item in res.email]),
        address=unique([item for res in results for item in res.address]),
    )


def fix_pii(pii: Pii) -> Pii:
    result = pii.copy()

    result.name = [name for name in pii.name if len(name) >= 2]

    def is_valid_rrn(rrn: str) -> bool:
        if '-' not in rrn:
            return len(rrn) == 13
        if len(rrn) != 14:
            return False
        parts = rrn.split('-')
        if len(parts) != 2:
            return False
        first, second = parts
        if len(first) != 6 or len(second) != 7:
            return False
        if second[0] not in ['1', '2', '3', '4']:
            return False
        return True

    result.rrn = [rrn for rrn in pii.rrn if is_valid_rrn(rrn)]

    return result


async def detect_pii(text: str) -> LlmResponse:
    text = text.strip().replace("\n", "")
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}/{len(chunks)}: {chunk}")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0

    async def process_chunk(i: int, chunk: str) -> Pii:
        nonlocal total_prompt_tokens, total_completion_tokens, total_calls
        print(f"Analyzing Chunk {i + 1}/{len(chunks)}...")

        cb = OpenAICallbackHandler()
        result = await chain.ainvoke({"text": chunk}, config={"callbacks": [cb]})
        fixed = fix_pii(result)

        total_prompt_tokens += cb.prompt_tokens
        total_completion_tokens += cb.completion_tokens
        total_calls += 1

        print(f"PII of Chunk {i + 1}/{len(chunks)}: {fixed}")
        return fixed

    if llm_model.model.startswith('gpt'):
        results = await asyncio.gather(*[process_chunk(i, chunk) for i, chunk in enumerate(chunks)])
    else:
        results = []
        for i, chunk in enumerate(chunks):
            result = await process_chunk(i, chunk)
            results.append(result)
    merged = merge_results(results)
    print("Results:")
    print(f"- PII: {merged}")
    print(f"- Input token: {total_prompt_tokens}")
    print(f"- Output token: {total_completion_tokens}")
    print(f"- Calls: {total_calls}")

    return LlmResponse(
        pii=merged,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        calls=total_calls
    )