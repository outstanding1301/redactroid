from pydantic import BaseModel


class Pii(BaseModel):
    name: list[str]
    phone: list[str]
    rrn: list[str]
    email: list[str]
    address: list[str]


class LlmResponse(BaseModel):
    pii: Pii
    prompt_tokens: int
    completion_tokens: int
    calls: int
