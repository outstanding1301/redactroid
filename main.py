from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

import pdf_service
from models import Pii, LlmResponse
# from pii_detector import detect_pii
from graph import detect_pii

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}."
        )
    contents = await file.read()
    if file.content_type == "text/plain":
        text = contents.decode("utf-8")
    else:
        text = pdf_service.extract_text(contents)

    pii_response = await detect_pii(text)
    # pii_response = LlmResponse(
    #     pii=Pii(
    #         name=["정 태 희", "이은섭"],
    #         phone=["042-480-3042", "042-480-3043", "042-480-3020"],
    #         rrn=[],
    #         email=['kska@tjcci.or.kr', 'eslee_dj@korcham.net'],
    #         address=['주한UAE 대사관', '아부다비', '호텔 인터시티 5층 사파이어홀', '대전상공회의소', '대전광역시 서구 대덕대로 176번길 51', '대전시'],
    #     ),
    #     prompt_tokens=5501,
    #     completion_tokens=243,
    #     calls=3,
    # )

    return JSONResponse(
        jsonable_encoder(pii_response.pii),
    )


@app.post("/redact")
async def redact(file: UploadFile = File(...),
                 name=Form(), phone=Form(), rrn=Form(), email=Form(), address=Form()) -> Response:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}."
        )

    contents = await file.read()
    pii = Pii(
        name=name.split(","),
        phone=phone.split(","),
        rrn=rrn.split(","),
        email=email.split(","),
        address=address.split(","),
    )
    res = pdf_service.redact(contents, pii)
    return Response(
        res,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=redacted.pdf",
        }
    )
