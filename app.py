from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

# --- Updated LangChain Core & Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Agentic Student Wellness API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)

# Setup Template Rendering for the Dashboard
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    # Fixed Starlette rendering signature
    return templates.TemplateResponse(request=request, name="index.html")

# Load the comprehensive pipeline model
with open("artifacts/stress_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Pydantic Schemas ---
class StudentMetrics(BaseModel):
    Student_Type: str          
    Sleep_Hours: float
    Study_Hours: float
    Social_Media_Hours: float
    Attendance: float
    Exam_Pressure: float        
    Family_Support: float       
    Month: float                

class ChatTurn(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    message: str
    history: List[ChatTurn]
    context: dict

# --- Agentic Handoff Functions ---
def get_initial_counselor_response(data: dict):
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a warm, highly empathetic student wellness counselor. "
            "A student has been flagged with high stress levels based on their lifestyle inputs:\n"
            "- Student Type: {Student_Type}\n"
            "- Sleep: {Sleep_Hours} hours\n"
            "- Study: {Study_Hours} hours\n"
            "- Exam Pressure: {Exam_Pressure}/10\n"
            "- Attendance: {Attendance}%\n"
            "Gently validate their current situation without being clinical. "
            "Offer one actionable, comforting suggestion based on their metrics, "
            "and ask an open, comforting question to guide them into a supportive conversation. Keep your response brief."
        )),
        ("user", "Hello counselor, I've been feeling pretty overwhelmed lately.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(data)

# --- API Endpoints ---
@app.post("/analyze")
async def analyze_stress_metrics(metrics: StudentMetrics):
    try:
        input_data = pd.DataFrame([{
            "Student_Type": metrics.Student_Type,
            "Sleep_Hours": metrics.Sleep_Hours,
            "Study_Hours": metrics.Study_Hours,
            "Social_Media_Hours": metrics.Social_Media_Hours,
            "Attendance": metrics.Attendance,
            "Exam_Pressure": metrics.Exam_Pressure,
            "Family_Support": metrics.Family_Support,
            "Month": metrics.Month
        }])
        
        prediction = model.predict(input_data)[0]
        
        if prediction == 0:
            return {
                "status": "low_stress",
                "message": "Your metrics show a strong, healthy balance! Keep prioritizing your well-being, getting enough rest, and pacing yourself."
            }
        else:
            student_context = metrics.model_dump()
            agent_reply = get_initial_counselor_response(student_context)
            return {
                "status": "high_stress",
                "message": agent_reply
            }
            
    except Exception as e:
        import traceback
        print("❌ DETECTED ENDPOINT CRASH ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def conversational_chat(payload: ChatPayload):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)
        
        # Format the dictionary into a clean bulleted list to avoid LangChain curly-brace errors
        context_str = "\n".join([f"- {key}: {value}" for key, value in payload.context.items()])
        
        sys_msg = (
            "You are an empathetic, professional student wellness counselor. "
            "Keep your answers concise, conversational, and warm. "
            "Here is the student's telemetry context to personalize your advice:\n"
            f"{context_str}"
        )
        
        messages = [("system", sys_msg)]
        
        for turn in payload.history:
            messages.append((turn.role, turn.content))
            
        messages.append(("user", payload.message))
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({})
        return {"reply": response}
        
    except Exception as e:
        import traceback
        print("❌ CHAT ENDPOINT CRASH:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))