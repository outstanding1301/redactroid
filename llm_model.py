import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = os.getenv('LLM_MODEL')
print(f"Model: {model}")
if model.startswith('gpt'):
    llm = ChatOpenAI(model=model, temperature=0)
elif model.startswith('gemini'):
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    