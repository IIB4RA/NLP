from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from router import classify_intent
from retriever import get_relevant_context
from generator import generate_rag_response

app = FastAPI(title="University RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model to accept query and history
class ChatRequest(BaseModel):
    query: str
    history: str = ""

@app.get("/")
async def root():
    return {"status": "healthy", "message": "University RAG API is running"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_question = request.query.strip()
    chat_history = request.history.strip()
    
    intent = classify_intent(user_question)
    
    if intent == "GREETING":
        response_text = "Welcome! I am your smart academic assistant. You can ask me about credit hour fees, disciplinary regulations, and admission or registration requirements. How can I help you today?"
    
    elif intent == "OFF_TOPIC":
        response_text = "Sorry, I am dedicated to answering questions related to university regulations and fees only. Please ask a question related to academic policies so I can assist you."

    elif intent == "UNIVERSITY_REGULATIONS":
        # Pass the combined history and query to the retriever for better context search
        combined_search_query = f"{chat_history} {user_question}"
        context = get_relevant_context(user_question)
        
        # Pass the history to the generator so it remembers the conversation
        response_text = generate_rag_response(user_question, context, chat_history)
        
    return {
        "intent": intent,
        "response": response_text
    }