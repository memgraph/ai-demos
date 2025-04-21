from sentence_transformers import SentenceTransformer
from typing import List

model_name = "all-mpnet-base-v2"

class EmbeddingGenerator:
    def __init__(self):
        self._model = SentenceTransformer(model_name)
    
    def get_embeddings(self, paragraphs: List[str]):
        embeddings = self._model.encode(paragraphs, convert_to_numpy=True)
        return embeddings

    def get_question_embedding(self, query: str):
        query_vector = self._model.encode(query).tolist()
        return query_vector
