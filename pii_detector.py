from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv

from models import Pii

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate.from_file('prompts/pii_detector.txt', encoding='utf-8')
output_parser = PydanticOutputParser(pydantic_object=Pii)

chain = prompt | llm | output_parser


def detect_pii(text: str) -> Pii:
    print(text)
    result = chain.invoke({"text": text})
    print(result)
    return result