import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import settings

qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

print("Loading Embedding Model (multilingual-e5-large)...")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Loading Reranker Model (bge-reranker-large)...")
reranker_name = "BAAI/bge-reranker-large"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
reranker_model.eval()


def get_relevant_context(
    user_query: str,
    collection_name: str,
    initial_top_k: int = 20,
    final_top_k: int = 5,
) -> str:
    """
    يبحث في collection محددة ويرجع أفضل chunks بعد الـ reranking.
    بدون score_threshold عشان ما نحذف chunks صح.
    """
    try:
        formatted_query = f"query: {user_query}"
        query_vector = embedding_model.encode(formatted_query).tolist()

        search_results = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=initial_top_k
        ).points

        if not search_results:
            print(f"[RETRIEVER] No results in {collection_name}")
            return ""

        candidate_chunks = [
            hit.payload.get("searchable_text", "")
            for hit in search_results
            if hit.payload.get("searchable_text")
        ]

        if not candidate_chunks:
            return ""

        # Reranking
        pairs = [[user_query, chunk] for chunk in candidate_chunks]
        with torch.no_grad():
            inputs = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            scores = reranker_model(**inputs).logits.view(-1).tolist()

        # خذ أفضل N بدون threshold — الـ generator هو اللي يقرر إذا الداتا كافية أم لا
        scored_chunks = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
        best_chunks = [chunk for chunk, score in scored_chunks[:final_top_k]]

        print(f"[RETRIEVER] {collection_name}: returned {len(best_chunks)} chunks | top score: {scored_chunks[0][1]:.2f}")
        return "\n\n---\n\n".join(best_chunks)

    except Exception as e:
        print(f"[RETRIEVER ERROR] {collection_name}: {e}")
        return ""