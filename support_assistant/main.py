from fastapi import FastAPI

from .models import (
    AskRequest,
    AskResponse
)

from .graph import graph


app = FastAPI(
    title="Zepto Support Assistant"
)


@app.get("/")
def home():

    return {
        "message":
            "Zepto Support Assistant is running"
    }


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    result = graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0
        }
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )