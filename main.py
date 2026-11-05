import base64
import os

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(
    title="Unlimited OCR API",
    description="Railway gateway for Unlimited-OCR GPU inference",
    version="1.0.0",
)

GPU_API_URL = os.getenv("GPU_API_URL", "").rstrip("/")
GPU_API_KEY = os.getenv("GPU_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Unlimited-OCR")

MAX_FILE_SIZE = 15 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@app.get("/")
async def root():
    return {
        "service": "Unlimited OCR API",
        "status": "running",
        "gpu_configured": bool(GPU_API_URL),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "unlimited-ocr-gateway",
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not GPU_API_URL:
        raise HTTPException(
            status_code=503,
            detail="GPU service is not configured yet.",
        )

    content_type = file.content_type or ""

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG and WebP files are supported.",
        )

    file_data = await file.read()

    if not file_data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 15 MB.",
        )

    encoded_image = base64.b64encode(file_data).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "document parsing.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{content_type};"
                                f"base64,{encoded_image}"
                            )
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "stream": False,
        "skip_special_tokens": False,
        "images_config": {
            "image_mode": "gundam",
        },
    }

    headers = {
        "Content-Type": "application/json",
    }

    if GPU_API_KEY:
        headers["Authorization"] = f"Bearer {GPU_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=1200) as client:
            response = await client.post(
                f"{GPU_API_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="GPU OCR service timed out.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to GPU OCR service: {exc}",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "GPU service returned an error.",
                "upstream_status": response.status_code,
                "upstream_response": response.text[:1500],
            },
        )

    result = response.json()

    try:
        extracted_text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        extracted_text = None

    return {
        "success": True,
        "filename": file.filename,
        "text": extracted_text,
        "raw_response": result,
    }