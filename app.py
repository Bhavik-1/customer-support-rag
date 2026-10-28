# app.py
# Experiment 7: Production-oriented FastAPI application.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import logging
import time

# Import the existing RAG pipeline.
from support_rag import ask_rag

# Configure basic logging.
# Do not log API keys or private customer conversations.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("customer_support_api")

# Create the FastAPI application.
app = FastAPI(
    title="Customer Support RAG API",
    description="Experiment 7 using Ollama and FastAPI",
    version="1.0.0"
)


# Define and validate incoming requests.
class AskRequest(BaseModel):
    # Require a question of at most 1,000 characters.
    question: str = Field(
        min_length=1,
        max_length=1000
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        # Reject questions containing only whitespace.
        if not value.strip():
            raise ValueError("Question must not be blank.")

        return value.strip()


# Define the structure of the API response.
class AskResponse(BaseModel):
    answer: str
    response_time_ms: float
    status: Literal["success", "fallback"]


# Health endpoint.
@app.get("/health")
def health():
    # Report whether the API process is responding.
    return {
        "status": "healthy",
        "service": "customer_support_rag_api"
    }


# Main RAG endpoint.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    # Record the request start time.
    start_time = time.perf_counter()

    try:
        # Call the existing RAG pipeline.
        answer = ask_rag(request.question)

        # Calculate response time in milliseconds.
        latency = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        # Identify the fallback response.
        fallback_text = (
            "I do not have sufficient information in the "
            "knowledge base to answer this question."
        )

        status = (
            "fallback"
            if fallback_text in answer
            else "success"
        )

        # Log the outcome and response time.
        logger.info(
            "Request completed: status=%s latency_ms=%.2f",
            status,
            latency
        )

        # Return a structured JSON response.
        return AskResponse(
            answer=answer,
            response_time_ms=latency,
            status=status
        )

    except Exception:
        # Log internal details for debugging.
        logger.exception("RAG request failed.")

        # Return a generic error without exposing internal details.
        raise HTTPException(
            status_code=500,
            detail="The request could not be processed. Please try again."
        )
