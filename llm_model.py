import os

import torch
from langchain_huggingface import HuggingFacePipeline

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, pipeline, AutoTokenizer

load_dotenv()

model = os.getenv('LLM_MODEL')
print(f"Model: {model}")
if model.startswith('gpt'):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
        ),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    hf_pipeline = pipeline(
        task="text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=512,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
