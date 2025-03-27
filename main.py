from fastapi import FastAPI
from pydantic import BaseModel
from pii_detector import detect_pii

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/detect")
def detect(input_text: InputText):
    result = detect_pii(input_text.text)
    return {"pii": result}
