import openai
import gradio as gr
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="finale.csv", encoding="cp1252")
pages = loader.load()

embedding = OpenAIEmbeddings()
llm_name = "gpt-4o"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

print("Beginning")
vectordb = FAISS.from_documents(pages, embedding)
print('Database Loaded')

template_uz = """Use the following pieces of context to answer the question at the end.
Please write all of the information in uzbek language. If the user asks to find a disease by symptoms, firstly, give two possible diseases. They don't have to be from retriever.
Then you have to include precaution for all two possible diseases. The last thing you have to do is include Description for only first possible disease and include information about Doctor.

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT_uz = PromptTemplate(input_variables=["context", "question"], template=template_uz)

qa_chain_uz = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=False,
                                          chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_uz})

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    response = qa_chain_uz.run(query.question)
    return {"response": response}
