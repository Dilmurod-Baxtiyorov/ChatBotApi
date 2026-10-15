from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Simple Uzbek Medical Chat API")

# -------------------------
# Request Model
# -------------------------
class Query(BaseModel):
    question: str


# -------------------------
# Mock LLM Response Logic
# -------------------------
def get_mock_response(question: str) -> str:
    q = question.lower()
    
    # 1. Greeting trigger
    if any(word in q for word in ["hello", "salom", "hi", "assalomu"]):
        return "Hello! I am a medical advisor. What is bothering you today? Please describe your symptoms and I will try to help."
    
    # 2. Stomach pain trigger
    elif any(word in q for word in ["stomach", "pain", "og'riq", "qorin", "oshqozon"]):
        return """1. Based on your symptoms, the two most likely conditions are: Acute Gastritis or Indigestion (Dyspepsia).
2. Measures to take for both: Follow a mild diet, avoid spicy and greasy foods, and drink plenty of warm water.
3. Acute Gastritis is an inflammation of the stomach lining, often caused by poor diet, stress, or bacterial infection. It usually presents as upper abdominal pain, nausea, and bloating.
4. For this condition, you should consult a Gastroenterologist.

Please make sure to consult a medical professional for an official diagnosis."""
    
    # 3. Fallback for any other question
    else:
        return "I understand. To ensure proper diagnosis and treatment for your health, please consult a qualified doctor."


# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
async def chat(query: Query):
    # Faking the LLM call instantly
    response_content = get_mock_response(query.question)

    return {
        "response": response_content
    }
