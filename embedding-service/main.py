from __future__ import annotations
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedding Service")
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: str = "all-MiniLM-L6-v2"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    texts = [request.input] if isinstance(request.input, str) else request.input
    vectors = model.encode(texts).tolist()
    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=v, index=i)
            for i, v in enumerate(vectors)
        ],
        model=request.model,
        usage={"prompt_tokens": 0, "total_tokens": 0},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
