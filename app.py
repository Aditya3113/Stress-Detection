from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="Agentic Student Wellness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

with open("artifacts/stress_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

class StudentMetrics(BaseModel):
    Student_Type: str          
    Sleep_Hours: float
    Study_Hours: float
    Social_Media_Hours: float
    Attendance: float
    Exam_Pressure: float        
    Family_Support: float       
    Month: float                

def get_agent_counselor_response(data: dict):
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
            "Offer one actionable, comforting suggestion based on their metrics (e.g., if sleep is low, mention rest), "
            "and ask an open, comforting question to guide them into a supportive conversation. Keep your response brief."
        )),
        ("user", "Hello counselor, I've been feeling pretty overwhelmed lately.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(data)

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
            agent_reply = get_agent_counselor_response(student_context)
            return {
                "status": "high_stress",
                "message": agent_reply
            }
            
    except Exception as e:
        import traceback
        print("❌ DETECTED ENDPOINT CRASH ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))