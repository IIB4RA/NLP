import requests
from config import settings


def classify_intent(user_query: str) -> str:
    URL = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"

    prompt = f"""Classify the following user input into exactly ONE category:

- GREETING: Casual hello, goodbye, thanks, how are you, or any social chitchat.
- OFF_TOPIC: Questions about general knowledge, sports, politics, technology, coding, or anything completely unrelated to a university's rules, fees, regulations, or academic life.
- UNIVERSITY_REGULATIONS: Any question related to university regulations, academic rules, fees, registration, training, grades, GPA, student activities, scholarships, disciplinary rules, or anything about university life.

When in doubt → use UNIVERSITY_REGULATIONS (it's safer to try searching than to reject).

Output ONLY the category name. No punctuation, no explanation.

User Input: "{user_query}"
Category:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 10, "temperature": 0.0}
    }

    try:
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            res_data = response.json()
            if "candidates" in res_data and res_data["candidates"]:
                candidate = res_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    intent = candidate["content"]["parts"][0]["text"].strip().upper()
                    if intent in ["GREETING", "OFF_TOPIC", "UNIVERSITY_REGULATIONS"]:
                        return intent
    except Exception as e:
        print(f"[ROUTER ERROR] {e}")

    return "UNIVERSITY_REGULATIONS"