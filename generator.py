import requests
from config import settings

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"


def generate_rag_response(user_query: str, context: str, chat_history: str = "") -> str:
    prompt = f"""أنت مساعد أكاديمي متخصص في أنظمة وتعليمات الجامعة الأردنية.

## مهمتك:
الإجابة على سؤال الطالب بناءً حصرياً على "السياق المستخرج" المرفق أدناه.

## قواعد صارمة:

1. **الأمانة في النقل:** استخدم فقط ما ورد في السياق. لا تضف معلومات من عندك أبداً.

2. **التعامل مع تعدد الأنظمة:**
   إذا وجدت في السياق أنظمة مختلفة (بكالوريوس / ماجستير / كليات مختلفة)، اعرضها منظمة هكذا:
   **للبكالوريوس:** ...
   **للدراسات العليا:** ...
   لا تدمجها ولا تخلط بينها.

3. **إذا السياق يحتوي الإجابة:** أجب مباشرة ولا تطلب توضيحاً.

4. **إذا لم تجد إجابة في السياق:** قل فقط:
   "عذراً، لم تتضمن اللوائح المتاحة حالياً إجابة دقيقة على هذا السؤال. يُنصح بالتواصل مع الدائرة المختصة."

5. **اللغة:** الإجابة دائماً بالعربية الفصحى الواضحة.

6. **التنسيق:** استخدم نقاط أو أرقام. لا تكتب فقرات طويلة متراصة.

7. **الإيجاز الذكي:** لا تعدد كل المواد — ركز على ما يجيب سؤال الطالب فعلاً.

---
## السياق المستخرج:
{context}

---
## المحادثة السابقة:
{chat_history if chat_history else "لا يوجد سياق سابق"}

---
## سؤال الطالب:
{user_query}

## الإجابة:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 6144,
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"خطأ في الاتصال: {response.status_code}"
    except Exception as e:
        return f"خطأ في توليد الإجابة: {e}"