import requests
from config import settings

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"

def generate_rag_response(user_query: str, context: str, chat_history: str = "") -> str:
    prompt = f"""You are a smart academic assistant and an expert in university rules and regulations. Your task is to answer the student's question based strictly on the provided extracted context.

Strict Instructions:
1. Rely entirely on the text provided in the "Extracted Context" section to form your answer.
2. If the retrieved context shows that the rules differ depending on the faculty or major, you MUST end your response with an open-ended question asking the student to specify their major. DO NOT list the specific majors found in the text. Simply ask a general question like, "Could you please specify your faculty or major so I can provide the exact requirements?"
3. If you cannot find the direct answer within the provided context, state clearly: "Sorry, I could not find the details regarding this information in the currently available regulations."

Previous Conversation Context:
{chat_history}

Extracted Context from University Regulations:
{context}

Student's Current Question:
{user_query}

Direct Answer in the language which the user uses:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"Connection Error: {response.status_code}"
    except Exception as e:
        return f"Generation Error: {e}"