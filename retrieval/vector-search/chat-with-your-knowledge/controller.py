from embeddings import EmbeddingGenerator
from memgraph_storage import MemgraphStorage
from dotenv import load_dotenv
from openai import OpenAI
from wikipedia_processor import WikipediaProcessor
from wikipedia_detailed_processor import DetailedWikipediaProcessor
import os
import json
import re
from typing import List

load_dotenv()

def get_ks_storage():
    return MemgraphStorage()


def sanitize_category(category: str) -> str:
    # Converts label to valid Cypher identifier (e.g., no spaces or special chars)
    return re.sub(r"[^a-zA-Z0-9_]", "_", category)


def extract_json(text: str) -> str:
    """Extract a JSON string from Markdown-style code blocks or plain output."""
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class StorageController:
    def __init__(self):
        self._storage = get_ks_storage()
        self._embedding_generator = EmbeddingGenerator()
        self._wikipedia_processor = WikipediaProcessor()
        self._wikipedi_detailed_processor = DetailedWikipediaProcessor()

    def get_all_categories(self) -> List[str]:
        return self._storage.get_all_categories()

    def ingest_wikipedia(
        self, category, save_as_category, lang_prefix, mode="replace", method="quick", section_filter=None
    ):
        if len(category) == 0:
            return 0
        category = sanitize_category(category)

        if len(save_as_category) == 0:
            save_as_category = category
        save_as_category = sanitize_category(save_as_category)
        
        if method == "quick":
            paragraphs, embeddings = (
                self._wikipedia_processor.process_wikipedia_documents(
                    category, lang_prefix
                )
            )
        else:
        # Else detailed
            paragraphs, embeddings = (
                self._wikipedi_detailed_processor.process_detailed_sections(
                    category, lang_prefix, section_filter
                )
            )

        if len(paragraphs) == 0:
            return 0

        return self._storage.ingest_paragraphs(
            save_as_category, paragraphs, embeddings, lang_prefix, mode
        )

    def get_similar_documents(self, category: str, question: str, n: int) -> List[str]:
        category = sanitize_category(category)
        query_vector = self._embedding_generator.get_question_embedding(question)

        results = self._storage.get_similar_documents(category, query_vector, n)

        context = [result["content"] for result in results]
        return context

    def get_paragraph_ids(self, category: str) -> List[int]:
        category = sanitize_category(category)
        return self._storage.get_paragraph_ids(category)
    
    def get_all_paragraphs_from_category(self, category: str):
        category = sanitize_category(category)
        return self._storage.get_all_paragraphs(category)

    
    def ingest_custom_text(self, category, paragraph, lang_prefix="custom", mode="append"):
        category = sanitize_category(category)
        paragraphs = [paragraph.strip()]
        embeddings = self._embedding_generator.get_embeddings(paragraphs)
        return self._storage.ingest_paragraphs(category, paragraphs, embeddings, lang_prefix, mode)
    
    def delete_paragraph(self, category: str, paragraph_id: str):
        category = sanitize_category(category)
        return self._storage.delete_paragraph(category, paragraph_id)


class LLMController:
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._storage = get_ks_storage()

    def answer_question_based_on_excerpts(
        self, question: str, context: List[str], lang_prefix: str
    ) -> str:
        context_text = "\n\n".join(context)

        prompt = f"""
    Using only the information from the following Wikipedia excerpts, answer the question below in two parts:

    1. Provide a **brief answer** (1–2 sentences max) that directly and clearly addresses the question. Begin it with "Short answer:".
    2. Then write a **detailed, coherent paragraph** that explains the answer using strictly the content from the excerpts. Begin it with "Coherent answer:".

    **Do not use any external knowledge** or make assumptions. If the answer is not present in the excerpts, clearly state that the information is not available.

    Avoid repeating phrases or listing bullet points. The longer explanation should read as a natural, well-written summary suitable for someone unfamiliar with the topic.

    Respond in the following language: {lang_prefix}

    Question: "{question}"

    Excerpts:
    {context_text}

    Answer:
    """

        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = response.choices[0].message.content

        return answer

    def generate_quiz(
        self,
        category: str,
        number_of_questions: int,
        lang_prefix: str,
        better_explanation: str,
    ):
        category = sanitize_category(category)
        results = self._storage.sample_n_connected_paragraphs(
            category, number_of_questions
        )
        if not results:
            return None
        context = [r["content"] for r in results]

        if len(context) == 0:
            return None

        context_text = "\n\n".join(context)

        quiz_prompt = f"""
        You are a pub quiz master. Using only the information in the text below, generate {number_of_questions} fun and challenging quiz questions with their answers.
        Questions should be easy to medium challenging. You can use your external knowledge to judge whether the question is something that the pub quiz participants can know. 
        You can not use external knowledge to form your answers, they should be only formed from the below text information.

        In addition to the question and answer, also include a short **explanation** that clearly states where and how the answer was derived from the provided text. This explanation should help someone understand the context or logic behind the answer, based solely on the text.

        Respond in the following language: **{lang_prefix}**.

        Focus only on facts from the content itself (avoid questions about references, publications, or sources).
        
        Here is an additional instruction for you to focus on specific questions:
        {better_explanation}

        Return a valid JSON array of {number_of_questions} objects. Each object must contain:
        - "question": the quiz question
        - "difficulty": "easy", "medium", "hard"
        - "answer": the correct answer
        - "explanation": a concise 2–3 sentence explanation based solely on the text which offers any relevant information to the answer

        Example:
        [
        {{
            "question": "What year was Rome founded?",
            "difficulty": "Easy",
            "answer": "753 BC",
            "explanation": "The text states that Rome was founded in 753 BC by Romulus. This marks the beginning of Roman history according to tradition."
        }},
        ...
        ]

        Text:
        {context_text}
        """

        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": quiz_prompt}],
            temperature=0.7,
        )

        raw_output = response.choices[0].message.content.strip()
        cleaned_output = extract_json(raw_output)

        try:
            quiz = json.loads(cleaned_output)
            return quiz
        except json.JSONDecodeError as e:
            raise e
