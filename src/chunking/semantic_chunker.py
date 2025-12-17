import yaml, json, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import nltk
from pathlib import Path
# nltk.download('punkt')
# nltk.download('punkt_tab')

from typing import List

# from pdfminer.high_level import extract_text

# def load_pdf_text(self, pdf_path):
#     try:
#         return extract_text(pdf_path)
#     except:
#         raise RuntimeError("PDF extraction failed — use OCR version.")

class SemanticChunker:
    def __init__(self, cfg_path="config.yaml"):
        cfg = yaml.safe_load(open(cfg_path))
        self.model = SentenceTransformer(cfg['embedding_model'])
        self.buffer = cfg.get('buffer_size', 5)
        self.theta = cfg.get('theta', 0.75)
        self.max_tokens = cfg.get('max_tokens_chunk', 1024)
        self.subchunk_tokens = cfg.get('subchunk_tokens', 128)

    def load_pdf_text(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        with open("data/processed/full_text.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(pages))
        return "\n".join(pages)

    def sentence_split(self, text: str) -> List[str]:
        return nltk.sent_tokenize(text)

    def buffer_merge(self, sentences: List[str]) -> List[str]:
        # Merge neighbor sentences using buffer window
        merged = []
        n = len(sentences)
        for i in range(n):
            start = max(0, i - self.buffer)
            end = min(n, i + self.buffer + 1)
            merged.append(" ".join(sentences[start:end]))
        
        out = []
        prev = None
        for m in merged:
            if m != prev:
                out.append(m)
            prev = m
        return out

    def embed_sentences(self, sents: List[str]):
        return self.model.encode(sents, convert_to_numpy=True, show_progress_bar=False)

    def chunk_by_cos(self, sents: List[str]):
        merged = self.buffer_merge(sents)
        embeds = self.embed_sentences(merged)
        chunks = []
        cur = [merged[0]]
        for i in range(len(merged)-1):
            sim = cosine_similarity([embeds[i]], [embeds[i+1]])[0][0]
            if sim >= self.theta:
                cur.append(merged[i+1])
            else:
                chunks.append(" ".join(cur))
                cur = [merged[i+1]]
        chunks.append(" ".join(cur))
        return chunks

    def enforce_token_limits(self, chunk: str):
        
        words = chunk.split()
        if len(words) <= self.max_tokens:
            return [chunk]
        # split into overlapping sub-chunks of subchunk_tokens with overlap 128 tokens
        step = self.subchunk_tokens - 128 if self.subchunk_tokens > 128 else int(self.subchunk_tokens*0.75)
        out = []
        for i in range(0, len(words), step):
            part = words[i:i + self.subchunk_tokens]
            out.append(" ".join(part))
            if i + self.subchunk_tokens >= len(words):
                break
        return out

    def process_pdf_to_chunks(self, pdf_path: str, out_path="data/processed/chunks.json"):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        if not os.path.isdir(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
        text = self.load_pdf_text(pdf_path)
        sents = self.sentence_split(text)
        raw_chunks = self.chunk_by_cos(sents)
        final_chunks = []
        for ch in raw_chunks:
            final_chunks.extend(self.enforce_token_limits(ch))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        return final_chunks


if __name__ == "__main__":
    chunker = SemanticChunker("config.yaml")
    print("Loading and chunking PDF...")
    current_dir = Path.cwd()
    # print("Current directory:", current_dir)
    data_dir = ''
    if(current_dir / 'data').exists():
        pdf_path = str(current_dir / 'data' / 'Ambedkar_book.pdf')
        print("---Found PDF at:", pdf_path)
        data_dir = str(current_dir / 'data')
    else:
        print("data dir not found in current directory. Searching parent directories...")
    current_dir = [dir for dir in current_dir.iterdir() if dir.is_dir() and dir.name == 'data'][0]

    for parent in current_dir.parents:
        if (parent / 'data' ).exists() and (parent / 'myenv').exists():
            pdf_path = (parent / 'data' / 'Ambedkar_book.pdf')
            data_dir = str(parent / 'data')
            print("Found PDF at:", pdf_path)
            break
    print("PDF path to be used:", pdf_path)
    text = chunker.load_pdf_text(pdf_path)
    print("text size:", len(text))
    sents = chunker.sentence_split(text)
    print("Number of sentences:", len(sents))
    raw_chunks = chunker.chunk_by_cos(sents)
    print("Number of raw chunks:", len(raw_chunks))
    final_chunks = []
    for ch in raw_chunks:
        final_chunks.extend(chunker.enforce_token_limits(ch))
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    out_path = os.path.join(data_dir, "processed/chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved final chunks to {out_path}")

    # chunks = chunker.process_pdf_to_chunks(pdf_path, out_path=os.path.join(data_dir, "processed/chunks.json"))
    # print(f"Generated {len(chunks)} chunks.")

