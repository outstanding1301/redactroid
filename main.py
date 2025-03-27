from fastapi import FastAPI, File, UploadFile, HTTPException, Response

from pii_detector import detect_pii

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...), response: Response = None):
    if file.content_type != "text/plain":
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식입니다: {file.content_type}. text/plain만 허용됩니다."
        )

    contents = await file.read()
    text = contents.decode("utf-8")
    pii_response = await detect_pii(text)

    response.headers["redactroid_prompt_tokens"] = str(pii_response.prompt_tokens)
    response.headers["redactroid_completion_tokens"] = str(pii_response.completion_tokens)
    response.headers["redactroid_calls"] = str(pii_response.calls)
    return {"pii": pii_response.pii}
