import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import settings

qdrant = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
COLLECTION_NAME = "university_regulations"

print("Loading Stage 1: Embedding Model (multilingual-e5-large)...")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Loading Stage 2: Reranker Model (bge-reranker-large)...")
reranker_name = "BAAI/bge-reranker-large"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
reranker_model.eval()

def get_relevant_context(user_query: str, initial_top_k: int = 20, final_top_k: int = 8) -> str:
    try:
        formatted_query = f"query: {user_query}"
        query_vector = embedding_model.encode(formatted_query).tolist()

        search_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=initial_top_k
        ).points

        candidate_chunks = [hit.payload.get("searchable_text", "") for hit in search_results if hit.payload.get("searchable_text")]

        if not candidate_chunks:
            return ""

        pairs = [[user_query, chunk] for chunk in candidate_chunks]
        with torch.no_grad():
            inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = reranker_model(**inputs).logits.view(-1).tolist()

        scored_chunks = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
        best_chunks = [chunk for chunk, score in scored_chunks[:final_top_k]]
        
        final_context = "\n\n---\n\n".join(best_chunks)

        print(f"\n=== RERANKED CONTEXT (Top {final_top_k} out of {len(candidate_chunks)}) ===")
        print(final_context)
        print("================================================\n")

        return final_context

    except Exception as e:
        print(f"Retrieval Error: {e}")
        return ""