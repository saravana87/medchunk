import re
from typing import List, Dict
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np

@dataclass
class Chunk:
    text: str
    keywords: List[str] = field(default_factory=list)
    cluster_label: int = -1

class MedicalChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.chunk_size = 100
        self.keybert = KeyBERT(model=model_name)
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.clusterer = None

    def chunk_text(self, text: str):
        sentences = re.split(r'[\n\.!?]', text)
        buffer = []
        length = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            buffer.append(sent)
            length += len(sent)
            if length >= self.chunk_size:
                self.chunks.append(Chunk(text=' '.join(buffer)))
                buffer, length = [], 0
        if buffer:
            self.chunks.append(Chunk(text=' '.join(buffer)))

    def extract_keywords(self, top_n=3):
        for chunk in self.chunks:
            keywords = self.keybert.extract_keywords(chunk.text, top_n=top_n, stop_words='english')
            chunk.keywords = [kw for kw, _ in keywords]

    def cluster_keywords(self):
        all_keywords = list({kw for chunk in self.chunks for kw in chunk.keywords})
        embeddings = self.embedding_model.encode(all_keywords)
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True)
        labels = self.clusterer.fit_predict(embeddings)
        keyword_to_cluster = {kw: label for kw, label in zip(all_keywords, labels)}
        for chunk in self.chunks:
            cluster_votes = [keyword_to_cluster.get(kw, -1) for kw in chunk.keywords]
            cluster_votes = [c for c in cluster_votes if c != -1]
            if cluster_votes:
                chunk.cluster_label = max(set(cluster_votes), key=cluster_votes.count)

    def run(self, text: str):
        self.chunk_text(text)
        self.extract_keywords()
        self.cluster_keywords()
        return self.chunks

if __name__ == "__main__":
    doc = """
    The patient was diagnosed with diabetes and prescribed insulin.
    Blood sugar levels were monitored regularly.
    The doctor noted signs of hypertension and suggested lifestyle changes.
    Cholesterol was also elevated in the recent tests.
    """

    chunker = MedicalChunker()
    chunks = chunker.run(doc)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(" Text:", chunk.text)
        print(" Keywords:", chunk.keywords)
        print(" Cluster Label:", chunk.cluster_label)
        print()
