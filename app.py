from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

app = FastAPI(title="Agentic Wellness Handoff API")

# --- 1. Load Custom ML Artifacts ---
with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("artifacts/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Pydantic Models ---
class UserInput(BaseModel):
    text: str

# --- 2. The LangChain Chat Counselor ---
def get_counselor_response(user_text: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an empathetic, professional wellness assistant. The user's initial text indicated psychological stress. Validate their feelings gently, offer a very brief grounding exercise, and ask a warm, open-ended question to help them talk through it. Keep it concise and supportive. Do not give medical diagnoses."),
        ("user", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_text})

# --- 3. API Endpoints ---
@app.post("/analyze")
async def analyze_input(user_input: UserInput):
    try:
        # A. Vectorize the raw text
        text_vec = vectorizer.transform([user_input.text])
        
        # B. Predict using Random Forest
        prediction = model.predict(text_vec)[0]
        
        # C. The Architecture Handoff
        if prediction == 0:
            return {
                "status": "low_stress",
                "message": "It sounds like you're in a decent headspace right now! Keep up the healthy habits, and remember to take breaks.",
            }
        else:
            # Trigger the Agentic Counselor Handoff
            agent_reply = get_counselor_response(user_input.text)
            return {
                "status": "high_stress",
                "message": agent_reply,
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))