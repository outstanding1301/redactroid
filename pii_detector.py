import asyncio
import textwrap
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from models import Pii, LlmResponse

load_dotenv()

CHUNK_SIZE = 1000

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    chunks = split_text(text)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0

    async def process_chunk(i: int, chunk: str) -> Pii:
        nonlocal total_prompt_tokens, total_completion_tokens, total_calls
        print(f"청크 {i + 1}/{len(chunks)} 분석 중...")

        cb = OpenAICallbackHandler()
        result = await chain.ainvoke({"text": chunk}, config={"callbacks": [cb]})
        fixed = fix_pii(result)

        total_prompt_tokens += cb.prompt_tokens
        total_completion_tokens += cb.completion_tokens
        total_calls += 1

        print(fixed)
        return fixed

    results = await asyncio.gather(*[process_chunk(i, chunk) for i, chunk in enumerate(chunks)])
    merged = merge_results(results)

    return LlmResponse(
        pii=merged,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        calls=total_calls
    )