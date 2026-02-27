from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="Simple Uzbek Medical Chat API")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

SYSTEM_PROMPT = """Sen tibbiy maslahatchisan. Har qanday javobingni faqat o‘zbek tilida ber.
Agar foydalanuvchi simptomlar bo‘yicha kasallik so‘rasa:
1. Avval ikkita eng ehtimoliy kasallikni ayt (o‘zing bilganing bo‘yicha, hujjatlarsiz).
2. Har ikkalasiga ham qanday choralarni ko‘rish kerakligini yoz (precautions).
3. Faqat birinchi kasallik haqida batafsil tavsif (description) ber.
4. Shu kasallik bo‘yicha qaysi shifokorga murojaat qilish kerakligini ayt.

Javoblaringni aniq, tushunarli va foydali qil. Tibbiy maslahat o‘rniga shifokorga borishni tavsiya etishni unutmang."""

@app.post("/chat")
async def chat(question: str):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return {"response": response.content.strip()}


class Query(BaseModel):
    question: str
