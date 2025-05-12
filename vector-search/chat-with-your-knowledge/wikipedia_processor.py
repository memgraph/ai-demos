from llama_index.readers.wikipedia import WikipediaReader
from embeddings import EmbeddingGenerator

class WikipediaProcessor:
    def __init__(self):
        self._embeddings_generator = EmbeddingGenerator()

    def process_wikipedia_documents(self, category: str, language_prefix: str = ""):
        reader = WikipediaReader()
        documents = reader.load_data(pages=[category], lang_prefix=language_prefix)
        paragraphs = [p.strip() for doc in documents for p in doc.text.split('\n') if len(p.strip()) > 40]
        embeddings = self._embeddings_generator.get_embeddings(paragraphs)
        
        return paragraphs, embeddings

