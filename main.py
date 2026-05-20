import json
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from router import classify_intent
from retriever import get_relevant_context
from generator import generate_rag_response
from config import settings

app = FastAPI(title="University RAG Assistant API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: str = ""

# الـ collections الموجودة فعلاً في Qdrant
PRIMARY_COLLECTION = "university_regulations"   # 1207 points - الأساسية
SPECIFIC_COLLECTIONS = [
    "regulations_undergrad",   # 250 points
    "regulations_postgrad",    # 103 points
    "regulations_general",     # 568 points
]


def build_routing_decision(user_query: str, chat_history: str) -> dict:
    """
    يبني search_query كامل من السياق الكامل للمحادثة،
    ويقرر هل يحتاج collections إضافية بجانب الـ primary.
    """
    prompt = f"""You are an expert academic query analyzer for the University of Jordan RAG system.

Your job:
1. Read the FULL conversation history carefully
2. Understand what the user is REALLY asking — even if their latest message is short (like "AI", "medicine", "what are the goals")
3. Build ONE complete, specific Arabic search query that combines ALL relevant context from history + current message
4. Decide if additional specific collections are needed (besides the main one)
5. Decide if the query is truly ambiguous and needs clarification

Collections (always search primary first — only add specific ones if needed):
- PRIMARY (always searched): "university_regulations" — contains ALL university data
- "regulations_undergrad": Only add if question is specifically about Bachelor programs
- "regulations_postgrad": Only add if question is specifically about Master/PhD programs
- "regulations_general": Only add if question is specifically about student activities/union/conduct

CRITICAL RULES:
- NEVER use the user's short message alone as the search_query
- search_query MUST be a complete Arabic sentence/phrase (not keywords)
- needs_clarification = true ONLY if degree level is completely unknown AND it fundamentally changes the answer
- If topic is general (applies to all students) → needs_clarification = false

Respond ONLY with valid JSON, no extra text:
{{
    "search_query": "complete Arabic search query",
    "extra_collections": [],
    "needs_clarification": false,
    "clarification_question": null
}}

---
Conversation History:
{chat_history if chat_history else "No previous conversation"}

Latest User Message: {user_query}

JSON:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 250, "temperature": 0.0}
    }

    try:
        URL = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            raw = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            # Validate and sanitize
            if not result.get("search_query"):
                result["search_query"] = user_query
            if not isinstance(result.get("extra_collections"), list):
                result["extra_collections"] = []
            # Only allow known collections
            result["extra_collections"] = [
                c for c in result["extra_collections"]
                if c in SPECIFIC_COLLECTIONS
            ]
            if "needs_clarification" not in result:
                result["needs_clarification"] = False
            if "clarification_question" not in result:
                result["clarification_question"] = None

            return result

    except Exception as e:
        print(f"[ERROR] Routing failed: {e}")

    # Safe fallback — search everything
    return {
        "search_query": user_query,
        "extra_collections": [],
        "needs_clarification": False,
        "clarification_question": None
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_question = request.query.strip()
    chat_history = request.history.strip()

    # Step 1: Basic intent classification
    intent = classify_intent(user_question)
    print(f"\n[LOG] Intent: {intent}")

    if intent == "GREETING":
        return {
            "response": "مرحباً بك! أنا مساعدك الأكاديمي في الجامعة الأردنية. كيف يمكنني مساعدتك اليوم؟",
            "intent": intent
        }
    elif intent == "OFF_TOPIC":
        return {
            "response": "عذراً، أنا مخصص للإجابة على الأسئلة المتعلقة بأنظمة وتعليمات الجامعة الأردنية فقط.",
            "intent": intent
        }

    # Step 2: Smart routing
    routing = build_routing_decision(user_question, chat_history)
    smart_query       = routing["search_query"]
    extra_collections = routing["extra_collections"]
    needs_clarification   = routing["needs_clarification"]
    clarification_question = routing["clarification_question"]

    print(f"[LOG] Smart Query: {smart_query}")
    print(f"[LOG] Extra Collections: {extra_collections}")
    print(f"[LOG] Needs Clarification: {needs_clarification}")

    # Step 3: Ask for clarification if truly needed
    if needs_clarification and clarification_question:
        return {
            "response": clarification_question,
            "intent": intent,
            "debug": {"reason": "needs_clarification", "query_built": smart_query}
        }

    # Step 4: Always search PRIMARY collection first
    all_contexts = []

    primary_context = get_relevant_context(smart_query, collection_name=PRIMARY_COLLECTION)
    if primary_context:
        all_contexts.append(primary_context)
        print(f"[LOG] Primary collection returned results")
    else:
        print(f"[LOG] Primary collection returned nothing — trying specific collections")

    # Step 5: Search extra specific collections
    for collection in extra_collections:
        context = get_relevant_context(smart_query, collection_name=collection)
        if context:
            all_contexts.append(context)
            print(f"[LOG] Extra collection {collection} returned results")

    # Step 6: If primary returned nothing, search ALL specific collections as fallback
    if not all_contexts:
        print(f"[LOG] Fallback: searching all specific collections")
        for collection in SPECIFIC_COLLECTIONS:
            context = get_relevant_context(smart_query, collection_name=collection)
            if context:
                all_contexts.append(context)

    merged_context = "\n\n---\n\n".join(all_contexts)

    if not merged_context:
        return {
            "response": "عذراً، لم أتمكن من العثور على معلومات كافية حول هذا الموضوع في اللوائح المتاحة. يُرجى التواصل مع الجهة المختصة في الجامعة.",
            "intent": intent,
            "debug": {
                "collections_searched": [PRIMARY_COLLECTION] + extra_collections,
                "query": smart_query
            }
        }

    # Step 7: Generate final answer
    response_text = generate_rag_response(smart_query, merged_context, chat_history)

    return {
        "intent": intent,
        "response": response_text,
        "debug": {
            "primary_collection": PRIMARY_COLLECTION,
            "extra_collections": extra_collections,
            "smart_query": smart_query
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)