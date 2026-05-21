import base64
import logging
import httpx
import fitz 
import docx
import io


from config import settings

logger = logging.getLogger(__name__)

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
)

SUPPORTED_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# Helpers
def _encode_image_bytes(image_bytes: bytes, mime_type: str) -> dict:
    # Returns a Gemini inline_data part for an image.
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }
    }


async def _extract_from_image(image_bytes: bytes, mime_type: str) -> str:
    # Send image to Gemini and extract all text.
    payload = {
        "contents": [{
            "parts": [
                _encode_image_bytes(image_bytes, mime_type),
                {"text": (
                    "استخرج كل النص الموجود في هذه الصورة كما هو بدون أي تعديل. "
                    "إذا لم يوجد نص، اكتب: [لا يوجد نص في الصورة]"
                )},
            ]
        }],
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.0},
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GEMINI_URL, json=payload)

    if resp.status_code != 200:
        raise ValueError(f"Gemini error: HTTP {resp.status_code} — {resp.text}")

    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _extract_from_pdf(file_bytes: bytes) -> str:
    # Extract text from PDF.
    # If the PDF has selectable text → extract directly .
    # If it's a scanned PDF → rasterize each page and send to Gemini .
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []

    for page in doc:
        text = page.get_text().strip()
        pages_text.append(text)

    full_text = "\n".join(pages_text).strip()

    # If we got meaningful text, return it directly
    if len(full_text) > 50:
        logger.info(f"[VISION] PDF: extracted {len(full_text)} chars via text layer")
        return full_text

    # Otherwise it's scanned — return page images for Vision processing
    logger.info("[VISION] PDF: no text layer found, flagging for OCR")
    return ""


def _extract_from_docx(file_bytes: bytes) -> str:
    # Extract all paragraph text from a Word document.
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    logger.info(f"[VISION] DOCX: extracted {len(text)} chars")
    return text



# Public functions
async def extract_text_from_file(
    file_bytes: bytes,
    mime_type: str,
    filename: str = "",
) -> str:
    """
    Main entry point. Accepts file bytes + MIME type, returns extracted text.
    Raises ValueError for unsupported types or extraction failures.
    """
    mime_type = mime_type.lower().strip()

    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"نوع الملف '{mime_type}' غير مدعوم. "
            f"الأنواع المدعومة: صور (jpg/png)، PDF، Word (.docx)"
        )

    # --- Image ---
    if mime_type in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        logger.info(f"[VISION] Processing image: {filename}")
        return await _extract_from_image(file_bytes, mime_type)

    # --- PDF ---
    if mime_type == "application/pdf":
        logger.info(f"[VISION] Processing PDF: {filename}")
        text = _extract_from_pdf(file_bytes)

        if text:
            return text

        # Scanned PDF — rasterize page 0 and send to Vision
        # (extend to all pages if needed)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        logger.info("[VISION] Scanned PDF — sending page image to Gemini Vision")
        return await _extract_from_image(img_bytes, "image/png")

    # --- Word ---
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        logger.info(f"[VISION] Processing DOCX: {filename}")
        return _extract_from_docx(file_bytes)

    raise ValueError("نوع ملف غير معروف.")


async def extract_question_from_text(raw_text: str) -> str:
    """
    If the extracted text is long/complex (e.g. a full document),
    use Gemini to identify what question the student is actually asking.
    If the text is short and clearly a question, return it as-is.
    """
    # Short text — treat directly as the query
    if len(raw_text) < 300:
        return raw_text.strip()

    prompt = f"""النص التالي مستخرج من ملف رفعه طالب جامعي.
حدد ما هو السؤال أو الطلب الذي يريد الطالب الاستفسار عنه.
أعد صياغته كسؤال واضح ومحدد بالعربية.
إذا كان النص يحتوي أسئلة متعددة، اجمعها في سؤال واحد شامل.
لا تضف أي معلومات من عندك.

النص:
{raw_text[:2000]}

السؤال:"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 200, "temperature": 0.0},
    }

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{settings.GENERATION_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
        )
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)

        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        logger.error(f"[VISION] Question extraction failed: {e}")

    # Fallback — use raw text directly
    return raw_text.strip()