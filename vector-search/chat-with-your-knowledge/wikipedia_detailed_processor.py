import wikipediaapi
from embeddings import EmbeddingGenerator


class DetailedWikipediaProcessor:
    def __init__(self):
        self._wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent='user_agent="WikiReaderBot/1.0 (mrdjen.josip@gmail.com)',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        self._embeddings_generator = EmbeddingGenerator()

    def _extract_paragraphs(self, section, min_length=40) -> list[str]:
        paragraphs = []
        if len(section.text.strip()) > min_length:
            paragraphs.extend(
                [
                    p.strip()
                    for p in section.text.split("\n")
                    if len(p.strip()) > min_length
                ]
            )
        for sub_section in section.sections:
            paragraphs.extend(self._extract_paragraphs(sub_section))
        return paragraphs

    def process_detailed_sections(
        self, category: str, language_prefix: str = "en", section_filter: str = None
    ):
        self._wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent='user_agent="WikiReaderBot/1.0 (mrdjen.josip@gmail.com)',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )
        page = self._wiki.page(category)

        if not page.exists():
            print(f"❌ Page '{category}' does not exist.")
            return [], []

        if section_filter:
            # Only grab sections that match the filter
            for section in page.sections:
                if section.title.lower() == section_filter.lower():
                    paragraphs = self._extract_paragraphs(section)
                    break
            else:
                print(f"⚠️ Section '{section_filter}' not found.")
                return [], []
        else:
            # Grab all content recursively
            paragraphs = self._extract_paragraphs(page)

        embeddings = self._embeddings_generator.get_embeddings(paragraphs)
        return paragraphs, embeddings
