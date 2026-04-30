from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="Simple Uzbek Medical Chat API")

# -------------------------
# Request Model
# -------------------------
class Query(BaseModel):
    question: str


# -------------------------
# Lazy LLM Initialization (IMPORTANT)
# -------------------------
llm = None

def get_llm():
    global llm
    if llm is None:
        print("Initializing LLM...")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
    return llm


# -------------------------
# System Prompt
# -------------------------
SYSTEM_PROMPT = """Sen tibbiy maslahatchisan. Har qanday javobingni faqat o‘zbek tilida ber.
Agar foydalanuvchi simptomlar bo‘yicha kasallik so‘rasa:
1. Avval ikkita eng ehtimoliy kasallikni ayt.
2. Har ikkalasiga ham qanday choralarni ko‘rish kerakligini yoz.
3. Faqat birinchi kasallik haqida batafsil tavsif ber.
4. Shu kasallik bo‘yicha qaysi shifokorga murojaat qilish kerakligini ayt.

Javoblaringni aniq, tushunarli va foydali qil. Shifokorga murojaat qilishni tavsiya etishni unutmang.
"""


# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat")
async def chat(query: Query):
    model = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query.question)
    ]

    response = model.invoke(messages)

    return {
        "response": response.content.strip()
    }
