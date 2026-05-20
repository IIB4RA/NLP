import json
import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from config import settings

# ─── Setup ───────────────────────────────────────────────────────────────────

qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

print("Loading embedding model...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large")

VECTOR_SIZE = 1024
BATCH_SIZE  = 64

# ─── Collection definitions ──────────────────────────────────────────────────

COLLECTIONS = {
    "university_regulations": "ALL chunks go here",
    "regulations_undergrad":  "Bachelor only",
    "regulations_postgrad":   "Postgrad only",
    "regulations_general":    "Student activities / union / services",
}

def ensure_collections():
    """يخلق الـ collection لو مش موجود، ولو موجود يتركه كما هو"""
    existing = {c.name for c in qdrant.get_collections().collections}
    for name in COLLECTIONS:
        if name in existing:
            info = qdrant.get_collection(name)
            print(f"  ✓ Already exists: {name} ({info.points_count} points) — keeping")
        else:
            qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"  ✓ Created new: {name}")

# ─── Routing logic ───────────────────────────────────────────────────────────

POSTGRAD_KEYWORDS = [
    "دراسات عليا", "ماجستير", "دكتوراه", "الدراسات العليا",
    "postgrad", "master", "phd"
]

UNDERGRAD_KEYWORDS = [
    "بكالوريوس", "undergraduate", "undergrad",
    "القبول والرسوم لطلبة البكالوريوس",
    "التسجيل والامتحانات لطلبة البكالوريوس",
    "التدريب",   # training is undergrad-heavy
]

GENERAL_KEYWORDS = [
    "اتحاد الطلبة", "الأندية", "الانشطة الطلابية",
    "خدمة المجتمع", "الخدمات الطلابية", "تأديب الطلبة",
    "التعلم والتعليم",
]

def get_specific_collection(chunk: dict) -> str:
    """يحدد الـ collection الخاص بالـ chunk بناءً على الـ metadata"""
    l2 = chunk["metadata"].get("level_2_topic", "")
    l3 = chunk["metadata"].get("level_3_document", "")
    text = f"{l2} {l3}".lower()

    for kw in POSTGRAD_KEYWORDS:
        if kw.lower() in text:
            return "regulations_postgrad"

    for kw in GENERAL_KEYWORDS:
        if kw.lower() in text:
            return "regulations_general"

    for kw in UNDERGRAD_KEYWORDS:
        if kw.lower() in text:
            return "regulations_undergrad"

    # default → undergrad (أغلب الداتا بكالوريوس)
    return "regulations_undergrad"

# ─── Indexing ────────────────────────────────────────────────────────────────

def index_chunks(chunks: list, source_file: str):
    print(f"\nIndexing {len(chunks)} chunks from {source_file}...")

    # Separate into collections
    routing: dict[str, list] = {name: [] for name in COLLECTIONS}

    for chunk in chunks:
        routing["university_regulations"].append(chunk)  # always
        specific = get_specific_collection(chunk)
        routing[specific].append(chunk)

    # Upload per collection
    for collection_name, col_chunks in routing.items():
        if not col_chunks:
            continue

        print(f"\n  → {collection_name}: {len(col_chunks)} chunks")
        texts = [c["searchable_text"] for c in col_chunks]

        # Embed in batches
        all_vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            prefixed = [f"passage: {t}" for t in batch]  # e5 prefix
            vecs = embedder.encode(prefixed, show_progress_bar=False).tolist()
            all_vectors.extend(vecs)
            print(f"    Embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")

        # Build points
        # Get current max ID in collection to avoid conflicts
        try:
            info = qdrant.get_collection(collection_name)
            start_id = info.points_count
        except:
            start_id = 0

        points = []
        for idx, (chunk, vector) in enumerate(zip(col_chunks, all_vectors)):
            point_id = start_id + idx
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "searchable_text": chunk["searchable_text"],
                    "content":         chunk.get("content", ""),
                    "article_number":  chunk.get("article_number", ""),
                    "level_1":         chunk["metadata"].get("level_1_main", ""),
                    "level_2":         chunk["metadata"].get("level_2_topic", ""),
                    "level_3":         chunk["metadata"].get("level_3_document", ""),
                    "level_4":         chunk["metadata"].get("level_4_sub_document", ""),
                    "source_file":     source_file,
                }
            ))

        # Upload in batches
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i+BATCH_SIZE]
            qdrant.upsert(collection_name=collection_name, points=batch)
            print(f"    Uploaded {min(i+BATCH_SIZE, len(points))}/{len(points)}", end="\r")

        print(f"    ✓ Done: {len(points)} points uploaded")
        time.sleep(0.5)  # avoid rate limits

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UJ RAG Indexer — Full Re-index")
    print("=" * 60)

    # Step 1: Recreate collections (wipes old data)
    print("\n[1] Ensuring collections exist...")
    ensure_collections()

    # Step 2: Load fixed chunks
    files = [
        "students_chunks_fixed.json",
        "students_chunks2_fixed.json",
    ]

    all_chunks = []
    for fname in files:
        try:
            with open(fname, encoding="utf-8") as f:
                data = json.load(f)
            all_chunks.append((data, fname))
            print(f"  Loaded {fname}: {len(data)} chunks")
        except FileNotFoundError:
            print(f"  ⚠ File not found: {fname} — skipping")

    if not all_chunks:
        print("No files to index. Run the fix script first.")
        exit(1)

    # Step 3: Index
    print("\n[2] Indexing...")
    for chunks, fname in all_chunks:
        index_chunks(chunks, fname)

    # Step 4: Summary
    print("\n[3] Final counts:")
    for name in COLLECTIONS:
        info = qdrant.get_collection(name)
        print(f"  {name}: {info.points_count} points")

    print("\n✓ Indexing complete!")
