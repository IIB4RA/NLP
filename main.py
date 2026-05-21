import json
import requests
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from router import classify_intent
from retriever import get_relevant_context
from generator import generate_rag_response
from config import settings

# Importing your new VLM functions
from vlm import extract_text_from_file, extract_question_from_text

# Initialize logger and global configuration constants
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10.0

app = FastAPI(title="University RAG Assistant API", version="5.0.0")

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

PRIMARY_COLLECTION = "university_regulations"
SPECIFIC_COLLECTIONS = [
    "regulations_undergrad",
    "regulations_postgrad",
    "regulations_general",
]


def build_routing_decision(user_query: str, chat_history: str) -> dict:
    prompt = f"""You are an expert academic query analyzer for the University of Jordan RAG system.

Your job:
1. Read the FULL conversation history carefully
2. Understand what the user is REALLY asking — even if their latest message is short
3. Build ONE complete, specific Arabic search query combining ALL context
4. Decide if additional specific collections are needed
5. Decide if the query is truly ambiguous

Collections:
- PRIMARY (always searched): "university_regulations" — contains ALL university data
- "regulations_undergrad": Only if specifically about Bachelor programs
- "regulations_postgrad": Only if specifically about Master/PhD programs
- "regulations_general": Only if specifically about student activities/union/conduct

RULES:
- NEVER use the user's short message alone as search_query
- search_query MUST be a complete Arabic sentence
- needs_clarification = true ONLY if degree level is unknown AND changes the answer fundamentally
- If topic is general → needs_clarification = false

Respond ONLY with valid JSON:
{{
    "search_query": "complete Arabic search query",
    "extra_collections": [],
    "needs_clarification": false,
    "clarification_question": null
}}

Conversation History:
{chat_history if chat_history else "No previous conversation"}

Latest User Message: {user_query}

JSON:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 250, "temperature": 0.0}
    }

    for attempt in range(3):
        try:
            URL = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
            response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                raw = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                result = json.loads(raw)

                if not result.get("search_query"):
                    result["search_query"] = user_query
                if not isinstance(result.get("extra_collections"), list):
                    result["extra_collections"] = []
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
            print(f"[ERROR] Routing attempt {attempt+1} failed: {e}")

    # Safe fallback
    return {
        "search_query": user_query,
        "extra_collections": [],
        "needs_clarification": False,
        "clarification_question": None
    }


def search_with_expansion(smart_query: str) -> list:

    queries = [
        smart_query,
        f"{smart_query} لجميع الطلبة",
        f"{smart_query} للدراسات العليا",
    ]

    all_contexts = []
    seen = set()

    for q in queries:
        result = get_relevant_context(q, collection_name=PRIMARY_COLLECTION)
        if result and result not in seen:
            all_contexts.append(result)
            seen.add(result)
            print(f"[LOG] Query '{q[:50]}...' → returned results")

    return all_contexts


async def run_core_rag_pipeline(user_question: str, chat_history: str) -> dict:
    # Step 1: Intent classification
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
    smart_query            = routing["search_query"]
    extra_collections      = routing["extra_collections"]
    needs_clarification    = routing["needs_clarification"]
    clarification_question = routing["clarification_question"]

    print(f"[LOG] Smart Query: {smart_query}")
    print(f"[LOG] Extra Collections: {extra_collections}")
    print(f"[LOG] Needs Clarification: {needs_clarification}")

    # Step 3: Clarification if needed
    if needs_clarification and clarification_question:
        return {
            "response": clarification_question,
            "intent": intent,
            "debug": {"reason": "needs_clarification", "query_built": smart_query}
        }

    # Step 4: Search PRIMARY with query expansion
    all_contexts = search_with_expansion(smart_query)

    # Step 5: Search extra specific collections
    for collection in extra_collections:
        context = get_relevant_context(smart_query, collection_name=collection)
        if context:
            all_contexts.append(context)
            print(f"[LOG] Extra collection {collection} returned results")

    # Step 6: Fallback — search ALL specific collections
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

    # Step 7: Generate answer
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
    


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_question = request.query.strip()
    chat_history = request.history.strip()

    return await run_core_rag_pipeline(user_question, chat_history)



@app.post("/api/chat/upload")
async def chat_upload_endpoint(
    file: UploadFile = File(...),
    history: str = Form(default=""),
    extra_query: str = Form(default=""),
):
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return {
            "response": f"عذراً، حجم الملف ({size_mb:.1f} MB) يتجاوز الحد المسموح ({MAX_FILE_SIZE_MB} MB).",
            "intent": "ERROR",
        }
 
    try:
        extracted_text = await extract_text_from_file(
            file_bytes=file_bytes,
            mime_type=file.content_type or "",
            filename=file.filename or "",
        )
    except ValueError as e:
        return {"response": str(e), "intent": "ERROR"}
    except Exception as e:
        logger.error(f"[UPLOAD] Extraction failed: {e}", exc_info=True)
        return {
            "response": "عذراً، حدث خطأ أثناء معالجة الملف. يرجى المحاولة مرة أخرى.",
            "intent": "ERROR",
        }
 
    if not extracted_text or extracted_text == "[لا يوجد نص في الصورة]":
        return {
            "response": "لم أتمكن من استخراج نص من هذا الملف. يرجى التأكد من أن الملف يحتوي على نص واضح.",
            "intent": "ERROR",
        }
 
    logger.info(f"[UPLOAD] Extracted {len(extracted_text)} chars from {file.filename}")
 
    combined_text = (
        f"{extra_query.strip()}\n\n{extracted_text}" if extra_query.strip()
        else extracted_text
    )
 
    user_question = await extract_question_from_text(combined_text)
    logger.info(f"[UPLOAD] Final question extracted from file context: {user_question}")
 
    # Run the extracted question through the exact same core pipeline asynchronously
    return await run_core_rag_pipeline(user_question=user_question, chat_history=history.strip())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)