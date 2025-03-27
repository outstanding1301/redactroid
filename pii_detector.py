from dotenv import load_dotenv
import re
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

template = """
다음 문장에서 아래 항목들에 해당하는 개인정보를 모두 찾아서 JSON 배열 형태로 반환해줘.

감지해야 할 항목:
- 이름 → name
- 전화번호 → phone
- 주민등록번호 → rrn
- 이메일 → email
- 주소 → address

출력 형식 (JSON 배열):
[
  {{
    "name": ["이름", ...],
    "phone": ["전화번호", ...],
    "rrn": ["주민등록번호", ...],
    "email": ["이메일", ...],
    "address": ["주소", ...]
  }},
  ...
]

* 주의:
- 개인정보가 포함되지 않은 항목은 생략하지 말고 []로 채워줘.
- JSON 외의 설명 없이 순수 JSON만 출력해줘.
- 하나의 문장에 여러 개인정보가 포함될 수 있음.

문장:
{text}
"""

prompt = PromptTemplate.from_template(template)
chain = prompt | llm

def extract_json(text: str) -> dict:
    try:
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        return json.loads(text)
    except Exception as e:
        return {"error": f"파싱 실패: {str(e)}", "raw": text}

def detect_pii(text: str) -> dict:
    result = chain.invoke({"text": text})
    return extract_json(result.content)
