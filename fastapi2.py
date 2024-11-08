from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from models import QueryLog
from database import SessionLocal
from utils3 import *
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os
from datetime import datetime

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize global conversation history (simulating session state)
conversation_history = {
    'responses': ["How can I assist you?"],
    'requests': []
}

# Initialize memory and LLM
if 'buffer_memory' not in globals():
    buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

ZHIPUAI_API_KEY = os.environ.get('ZHIPUAI_API_KEY')
llm = ChatZhipuAI(model="glm-4")

# Initialize conversation chain
system_msg_template = SystemMessagePromptTemplate.from_template(template="""用给你的文字尽可能真实的回答问题, 尽可能完整，长幅度的回答问题，不要回答单一句子，
如果答案不在以下的文字里, 说 '我不知道'.""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.post("/api/query")
async def query(request: QueryRequest, db: Session = Depends(get_db)):
    query = request.query
    user_id = request.user_id

    if not query:
        raise HTTPException(status_code=400, detail="没有问题")
    if not user_id:
        raise HTTPException(status_code=400, detail="没有openid")

    # Store the query in the database
    log = QueryLog(user_id=user_id, query=query, timestamp=datetime.utcnow())
    db.add(log)
    db.commit()

    # Process the query
    conversation_string = get_conversation_string()
    refined_query = query_refiner(conversation_string, query)
    context = find_match(refined_query)
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

    url = "http://docs.fitfithealth.com.cn/namecard/autumn_leaves_about_to_wither.jpg"

    return {"response": response, "pic":url, "code":0}
