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

# --- Hugging Face (replaces langchain_google_genai / ChatGoogleGenerativeAI) ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
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

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_counselor_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        task="text-generation",
        provider="auto",          # let HF route to whichever provider is hosting it
        max_new_tokens=300,
        temperature=0.6,
        repetition_penalty=1.15,
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=endpoint)


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


def get_initial_counselor_response(data: dict):
    llm = get_counselor_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a warm, highly empathetic student wellness counselor with real "
            "expertise in stress management techniques (CBT-style reframing, grounding "
            "exercises, study-load pacing). Avoid generic platitudes like 'take a break' "
            "with no specifics - always tie your suggestion to the numbers below.\n"
            "- Student Type: {Student_Type}\n"
            "- Sleep: {Sleep_Hours} hours\n"
            "- Study: {Study_Hours} hours\n"
            "- Exam Pressure: {Exam_Pressure}/10\n"
            "- Attendance: {Attendance}%\n"
            "Gently validate their current situation, offer one specific, actionable "
            "suggestion tied to their metrics, then ask an open, caring question. "
            "Keep it brief - 3-4 sentences."
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
            agent_reply = get_initial_counselor_response(student_context)
            return {
                "status": "high_stress",
                "message": agent_reply
            }

    except Exception as e:
        import traceback
        print("\u274c DETECTED ENDPOINT CRASH ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def conversational_chat(payload: ChatPayload):
    try:
        llm = get_counselor_llm()

        context_str = "\n".join([f"- {key}: {value}" for key, value in payload.context.items()])

        sys_msg = (
            "You are an empathetic, professional student wellness counselor with real "
            "expertise in stress management. Keep answers concise, conversational, warm, "
            "and specific - never generic reassurance. Here is the student's telemetry "
            "context to personalize your advice:\n"
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
        print("\u274c CHAT ENDPOINT CRASH:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
