"""
FastAPI backend for the Fake Review Detection Chrome extension.

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from predict import predict  # noqa: E402

app = FastAPI(
    title="Fake Review Detection API",
    description="Predicts whether a product review is fake (computer-generated) or real.",
    version="1.0.0",
)

# Chrome extensions call this from content scripts on arbitrary shopping sites
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_private_network(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_single(req: ReviewRequest):
    return predict(req.text)


@app.post("/predict/batch")
def predict_batch(req: BatchReviewRequest):
    return {"results": [predict(r) for r in req.reviews]}
