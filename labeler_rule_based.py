from keybert import KeyBERT
from typing import List
from dataclasses import dataclass, field
import re

@dataclass
class Chunk:
    text: str
    section: str = "UNKNOWN"
    labels: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

class Chunker:
    def __init__(self, min_length: int = 30):
        self.min_length = min_length

    def chunk_text(self, text: str) -> List[Chunk]:
        # Split into sections based on newlines or headers
        raw_chunks = re.split(r'\n{2,}|(?<=\.)\s+(?=[A-Z])', text)
        chunks = []
        for raw in raw_chunks:
            clean = raw.strip()
            if len(clean) >= self.min_length:
                chunks.append(Chunk(text=clean))
        return chunks

class AutoLabeler:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = KeyBERT(model=model_name)

    def extract_keywords(self, chunks: List[Chunk], top_n: int = 3, min_conf: float = 0.3) -> List[Chunk]:
        for chunk in chunks:
            results = self.model.extract_keywords(
                chunk.text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
            keywords = [kw for kw, score in results if score >= min_conf]
            chunk.keywords = keywords
        return chunks

    def label_from_keywords(self, chunks: List[Chunk]) -> List[Chunk]:
        for chunk in chunks:
            for keyword in chunk.keywords:
                label = self._guess_label(keyword)
                if label:
                    chunk.labels.append(label)
        return chunks

    def _guess_label(self, keyword: str) -> str:
        keyword = keyword.lower()
        if any(x in keyword for x in ["diabetes", "cancer", "asthma"]):
            return "disease"
        if any(x in keyword for x in ["insulin", "metformin"]):
            return "medication"
        if any(x in keyword for x in ["scan", "mri", "biopsy"]):
            return "procedure"
        return "unknown"

if __name__ == "__main__":
    sample_text = """
    The patient has a history of diabetes and was prescribed insulin daily. 
    Recent MRI scans revealed a possible lesion that needs further investigation. 
    Metformin was considered but not prescribed due to prior adverse reaction. 
    """

    print("== Chunking ==")
    chunker = Chunker(min_length=30)
    chunks = chunker.chunk_text(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:", chunk.text)

    print("\n== Keyword Extraction and Labeling ==")
    labeler = AutoLabeler()
    chunks = labeler.extract_keywords(chunks)
    chunks = labeler.label_from_keywords(chunks)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print("Text:", chunk.text)
        print("Keywords:", chunk.keywords)
        print("Labels:", chunk.labels)