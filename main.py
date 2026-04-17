from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client()

class Message(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    text: str
    history: List[Message] = []

SYSTEM_PROMPT = """
You are a friendly and helpful voice assistant for a food delivery app similar to Zomato. 
CRITICAL RULES:
1. Keep your answers brief (1-3 sentences maximum) because they are being read aloud by text-to-speech.
2. If a user wants to order food, suggest realistic options or confirm their order.
3. If a user has a complaint (e.g., missing food, cold food), be empathetic, apologize, and ask for their order number to initiate a refund or replacement.
"""


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(query: UserQuery):
   
    formatted_history = []
    for msg in query.history:
        gemini_role = "user" if msg.role == "user" else "model"
        formatted_history.append(
            {"role": gemini_role, "parts": [{"text": msg.content}]}
        )
    
   
    formatted_history.append(
        {"role": "user", "parts": [{"text": query.text}]}
    )

    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=formatted_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            )
        )
        return {"response": response.text}
        
    except Exception as e:
        print(f"\n--- DEBUG ERROR --- \n{e}\n-------------------\n")
        return {"response": "Sorry, I am having trouble connecting to my servers."}
