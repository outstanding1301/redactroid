from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models import Pii

load_dotenv()

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


def detect_pii(text: str) -> Pii:
    print(text)
    result = chain.invoke({"text": text})
    print(result)
    return result
