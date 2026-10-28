from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import logging
import time

from support_rag import ask_rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("customer_support_api")

app = FastAPI(
    title="Customer Support RAG System",
    description="Customer Support RAG using FastAPI and Gemini",
    version="1.1.0"
)


# -------------------- REQUEST MODEL --------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value.strip()


class AskResponse(BaseModel):
    answer: str
    response_time_ms: float
    status: Literal["success", "fallback"]


# -------------------- FRONTEND --------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Customer Support RAG</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6fb;
            color: #172033;
        }

        header {
            background: #ffffff;
            border-bottom: 1px solid #e1e5ed;
            padding: 20px 8%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            font-size: 20px;
            font-weight: bold;
            color: #263f86;
        }

        .badge {
            background: #e5f7eb;
            color: #237444;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
        }

        main {
            max-width: 850px;
            margin: 65px auto;
            padding: 0 20px;
        }

        .tag {
            color: #5269a6;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        h1 {
            font-size: 40px;
            margin: 15px 0;
        }

        .description {
            color: #697386;
            line-height: 1.7;
            font-size: 16px;
        }

        .card {
            background: white;
            margin-top: 30px;
            border: 1px solid #e0e5ee;
            border-radius: 14px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.04);
        }

        .card-title {
            font-size: 17px;
            font-weight: bold;
            margin-bottom: 22px;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            padding: 14px;
            border: 1px solid #d5dce8;
            border-radius: 9px;
            font-family: inherit;
            font-size: 15px;
            resize: vertical;
            outline: none;
        }

        textarea:focus {
            border-color: #5269a6;
            box-shadow: 0 0 0 3px #e9edfa;
        }

        .actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
        }

        .hint {
            color: #8790a0;
            font-size: 12px;
        }

        .ask-button {
            background: #263f86;
            color: white;
            border: none;
            padding: 13px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }

        .ask-button:hover {
            background: #1b306c;
        }

        .ask-button:disabled {
            opacity: 0.6;
            cursor: wait;
        }

        .examples {
            margin-top: 25px;
        }

        .examples p {
            font-size: 13px;
            color: #697386;
        }

        .example-button {
            background: white;
            border: 1px solid #dce2ed;
            border-radius: 20px;
            padding: 9px 12px;
            margin: 4px 3px 4px 0;
            cursor: pointer;
            font-size: 12px;
            color: #35435f;
        }

        .example-button:hover {
            background: #f0f3fb;
        }

        .answer {
            display: none;
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            background: #f6f8ff;
            border: 1px solid #dce4fa;
        }

        .answer h3 {
            margin-top: 0;
            font-size: 16px;
        }

        #answerText {
            line-height: 1.8;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font-size: 14px;
        }

        #answerMeta {
            color: #778198;
            font-size: 12px;
            margin-top: 15px;
        }

        .error {
            color: #b42318;
        }

        footer {
            text-align: center;
            color: #8991a0;
            font-size: 12px;
            padding: 25px;
            line-height: 1.8;
        }

        @media (max-width: 600px) {
            header {
                padding: 18px;
            }

            main {
                margin-top: 40px;
            }

            h1 {
                font-size: 31px;
            }

            .card {
                padding: 18px;
            }

            .actions {
                flex-direction: column;
                align-items: stretch;
            }

            .ask-button {
                width: 100%;
            }
        }
    </style>
</head>

<body>

<header>
    <div class="brand">Support Assistant</div>
    <div class="badge">● Service online</div>
</header>

<main>

    <div class="tag">Experiment 7 · Customer Support</div>

    <h1>How can we help you?</h1>

    <p class="description">
        Welcome to the Customer Support RAG System.
        Ask questions about shipping, returns, refunds, and customer
        support. The system retrieves relevant information from its
        knowledge base and generates an answer using AI.
    </p>

    <section class="card">

        <div class="card-title">Ask the Support Assistant</div>

        <form id="questionForm">

            <label for="question">Enter your question</label>

            <textarea
                id="question"
                maxlength="1000"
                placeholder="Example: How long does shipping take?"
                required
            ></textarea>

            <div class="actions">
                <span class="hint">
                    Avoid entering passwords or payment details.
                </span>

                <button class="ask-button" id="askButton" type="submit">
                    Ask Question →
                </button>
            </div>

        </form>

        <div class="examples">

            <p>Try one of these questions:</p>

            <button class="example-button"
                data-question="How long does shipping take?">
                Shipping time
            </button>

            <button class="example-button"
                data-question="What is the return policy?">
                Return policy
            </button>

            <button class="example-button"
                data-question="When will I receive my refund?">
                Refund information
            </button>

            <button class="example-button"
                data-question="How can I contact customer support?">
                Contact support
            </button>

        </div>

        <div class="answer" id="answerBox" aria-live="polite">

            <h3 id="answerHeading">Answer</h3>

            <div id="answerText"></div>

            <div id="answerMeta"></div>

        </div>

    </section>

</main>

<footer>
    Customer Support RAG System<br>
    FastAPI · Knowledge Retrieval · Gemini API
</footer>

<script>
    const form = document.getElementById("questionForm");
    const input = document.getElementById("question");
    const button = document.getElementById("askButton");

    const answerBox = document.getElementById("answerBox");
    const answerHeading = document.getElementById("answerHeading");
    const answerText = document.getElementById("answerText");
    const answerMeta = document.getElementById("answerMeta");

    // Fill the input when an example question is clicked.
    document.querySelectorAll(".example-button").forEach(function (item) {
        item.addEventListener("click", function () {
            input.value = item.dataset.question;
            input.focus();
        });
    });

    // Send the question to the existing FastAPI endpoint.
    form.addEventListener("submit", async function (event) {
        event.preventDefault();

        const question = input.value.trim();

        if (!question) {
            return;
        }

        button.disabled = true;
        button.textContent = "Please wait...";

        answerBox.style.display = "block";
        answerHeading.textContent = "Generating answer";
        answerText.className = "";
        answerText.textContent =
            "Searching the knowledge base and preparing your answer...";
        answerMeta.textContent = "";

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.detail || "The request could not be processed."
                );
            }

            answerHeading.textContent = "Answer";

            // Use textContent rather than innerHTML for safe display.
            answerText.textContent = data.answer;

            answerMeta.textContent =
                "Response time: " + data.response_time_ms +
                " ms | Status: " + data.status;

        } catch (error) {
            answerHeading.textContent = "Unable to get an answer";
            answerText.className = "error";
            answerText.textContent =
                error.message || "Please try again later.";
            answerMeta.textContent = "";
        } finally {
            button.disabled = false;
            button.textContent = "Ask Question →";
        }
    });
</script>

</body>
</html>
""")


# -------------------- HEALTH ENDPOINT --------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "customer_support_rag_api"
    }


# -------------------- ASK ENDPOINT --------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    start_time = time.perf_counter()

    try:
        answer = ask_rag(request.question)

        latency = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        fallback_text = (
            "I do not have sufficient information in the "
            "knowledge base to answer this question."
        )

        status = (
            "fallback"
            if fallback_text in answer
            else "success"
        )

        logger.info(
            "Request completed: status=%s latency_ms=%.2f",
            status,
            latency
        )

        return AskResponse(
            answer=answer,
            response_time_ms=latency,
            status=status
        )

    except Exception:
        logger.exception("RAG request failed.")

        raise HTTPException(
            status_code=500,
            detail="The request could not be processed. Please try again."
        )
