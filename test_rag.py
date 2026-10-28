# test_rag.py
# Test the RAG pipeline independently of FastAPI.

from support_rag import ask_rag

# Test a question that the knowledge base can answer.
question = "How long does shipping take?"

print("Question:", question)
print("Answer:", ask_rag(question))
