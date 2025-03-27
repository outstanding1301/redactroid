from pydantic import BaseModel


class Pii(BaseModel):
    name: list[str]
    phone: list[str]
    rrn: list[str]
    email: list[str]
    address: list[str]
