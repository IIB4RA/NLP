import requests
from config import settings

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"

def classify_intent(user_query: str) -> str:
    prompt = f"""Analyze the following user input and classify it into exactly one of these categories:
- GREETING: Casual hello, goodbye, thanks, or introductory chitchat.
- OFF_TOPIC: Questions about general knowledge, sports, politics, or coding completely unrelated to a university's rules, fees, or regulations.
- UNIVERSITY_REGULATIONS: Direct or indirect questions regarding university regulations, course fees, grades, registration, discipline systems, or point calculation math.

Output ONLY the category name string (GREETING, OFF_TOPIC, or UNIVERSITY_REGULATIONS). Do not include any other text or punctuation.

User Input: "{user_query}"
Category:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 25, "temperature": 0.0}
    }
    
    try:
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            res_data = response.json()
            
            # Safe checking of nested dictionary keys
            if "candidates" in res_data and res_data["candidates"]:
                candidate = res_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    intent = candidate["content"]["parts"][0]["text"].strip()
                    
                    if intent in ["GREETING", "OFF_TOPIC", "UNIVERSITY_REGULATIONS"]:
                        return intent
    except Exception as e:
        print(f"Router Error: {e}")
        pass
    
    return "UNIVERSITY_REGULATIONS" 