# support_rag.py
# Experiment 7: Customer Support RAG pipeline using Ollama.

import re
import requests

# Ollama configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5:0.5b"

# Sample customer-support knowledge base.
# Replace these documents with your Experiment 5/6 documents if available.
KNOWLEDGE_BASE = [
    {
        "topic": "Shipping",
        "content": (
            "Standard shipping takes 3 to 5 business days. "
            "Customers receive a tracking link after their order ships."
        )
    },
    {
        "topic": "Returns",
        "content": (
            "Customers can request a return within 30 days of delivery. "
            "The product must be unused and in its original packaging."
        )
    },
    {
        "topic": "Refunds",
        "content": (
            "Approved refunds are processed within 5 to 7 business days. "
            "Bank processing time may vary."
        )
    },
    {
        "topic": "Customer Support",
        "content": (
            "Customers can contact support at support@example.com. "
            "They should include their order number when contacting support."
        )
    }
]

# Common words that are not very useful for keyword retrieval.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "when", "where", "who", "why", "how",
    "do", "does", "did", "for", "of", "to", "in",
    "on", "and", "or", "my", "your", "it", "i",
    "can", "could", "would", "should", "be", "with",
    "tell", "me", "about", "please", "much", "long"
}


def tokenize(text):
    """Convert text to lowercase keywords."""
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())

    return {
        word for word in words
        if word not in STOP_WORDS
    }


def retrieve_documents(question, top_k=2):
    """
    Find relevant knowledge-base documents using keyword matching.
    This is a simple retriever for demonstration purposes.
    """

    question_words = tokenize(question)
    scored_documents = []

    for document in KNOWLEDGE_BASE:
        # Search both the topic and the document content.
        text = document["topic"] + " " + document["content"]
        document_words = tokenize(text)

        # Calculate keyword overlap.
        score = len(question_words.intersection(document_words))

        scored_documents.append((score, document))

    # Rank documents by their relevance scores.
    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # Return no documents when there is no meaningful match.
    if not scored_documents or scored_documents[0][0] == 0:
        return []

    return [
        document
        for score, document in scored_documents[:top_k]
        if score > 0
    ]


def ask_rag(question):
    """
    Retrieve relevant documents and generate an answer using Ollama.
    This function is called by the FastAPI application.
    """

    # Retrieve relevant knowledge-base documents.
    documents = retrieve_documents(question)

    # Avoid calling the LLM when retrieval finds no relevant information.
    if not documents:
        return (
            "I do not have sufficient information in the "
            "knowledge base to answer this question."
        )

    # Build the context from retrieved documents.
    context = "\n".join(
        f"{document['topic']}: {document['content']}"
        for document in documents
    )

    # Build the prompt for the language model.
    prompt = f"""
You are a customer-support assistant.

Answer the question using ONLY the information in the context.

If the answer is not present in the context, say:
"I do not have sufficient information in the knowledge base."

Do not invent company policies.

Context:
{context}

Customer question:
{question}

Answer:
"""

    # Send the prompt to the local Ollama API.
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    # Raise an exception if Ollama returns an HTTP error.
    response.raise_for_status()

    # Extract the generated answer.
    answer = response.json()["message"]["content"]

    return answer.strip()
