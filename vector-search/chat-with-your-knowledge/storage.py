from abc import ABC
from typing import List

class Storage(ABC):
    def get_all_categories(self):
        pass
    
    def ingest_category(self):
        pass
    
    def get_similar_documents(self, label: str, question: str, n: int):
        pass
    
    def get_paragraph_ids(self, category: str):
        pass
    
    def sample_n_connected_paragraphs(self, category: str, number_of_questions: int):
        pass

    def ingest_paragraphs(self, category: str, paragraphs: List[str], embeddings: List, lang_prefix: str, mode: str):
        pass