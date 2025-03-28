from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

import pdf_service
from pii_detector import detect_pii

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식입니다: {file.content_type}."
        )
    contents = await file.read()
    if file.content_type == "text/plain":
        text = contents.decode("utf-8")
    else:
        text = pdf_service.extract_text(contents)

    text = text.strip().replace("\n", "")
    print(f"Text: {text}")
    pii_response = await detect_pii(text)

    return JSONResponse(
        jsonable_encoder({"pii": pii_response.pii}),
        headers={
            "redactroid_prompt_tokens": str(pii_response.prompt_tokens),
            "redactroid_completion_tokens": str(pii_response.completion_tokens),
            "redactroid_calls": str(pii_response.calls),
        }
    )


@app.post("/redact")
async def redact(file: UploadFile = File(...)) -> Response:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 파일 형식입니다: {file.content_type}."
        )

    contents = await file.read()
    text = pdf_service.extract_text(contents)
    pii_response = await detect_pii(text)
    res = pdf_service.redact(contents, pii_response.pii)
    return Response(
        res,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=redacted.pdf",
            "redactroid_prompt_tokens": str(pii_response.prompt_tokens),
            "redactroid_completion_tokens": str(pii_response.completion_tokens),
            "redactroid_calls": str(pii_response.calls),
        }
    )
