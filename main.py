from fastapi import FastAPI, File, UploadFile, HTTPException

from pii_detector import detect_pii

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type != "text/plain":
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식입니다: {file.content_type}. text/plain만 허용됩니다."
        )

    contents = await file.read()
    text = contents.decode("utf-8")
    pii = detect_pii(text)
    return {"pii": pii}
