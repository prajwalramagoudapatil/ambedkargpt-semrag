import sys
from pathlib import Path

current_file_dir = Path(__file__).resolve().parent # directory of ambedkargpt.py = src/pipeline

src_directory = current_file_dir.parent # src directory

src_path_str = str(src_directory)
data_dir = src_directory.parent / 'data'

if src_directory.exists() and src_directory.is_dir():
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
        print(f"Added '{src_path_str}' to sys.path.")

print(f"Current sys.path entries: {sys.path[:3]}") # Show the first few entries

import json
from retrieval.local_search import LocalGraphRAG
from retrieval.global_search import GlobalGraphRAG
from llm.llm_client import generate_answer
import pickle

def interactive_demo():
    
    with open("data/processed/chunks.json", 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    communities = pickle.load(open("data/processed/communities.pkl","rb"))
    local = LocalGraphRAG(graph_path="data/processed/knowledge_graph.pkl")
    globalr = GlobalGraphRAG(graph_path="data/processed/knowledge_graph.pkl", communities=communities)
    while True:
        q = input("Question (or 'exit')> ")
        if q.strip().lower() in ("exit","quit", 'q', 'e'): break
        local_results = local.search(q, chunks)
        global_results = globalr.search(q, chunks)
        ans = generate_answer(q, local_results, global_results)
        print("=== ANSWER ===")
        print(ans)
        print("===SOME LOCAL SOURCES ===")
        for r in local_results[:5]:
            print(f"[CHUNK {r['chunk_idx']}] {r['chunk_text'][:200]}...")

if __name__ == "__main__":
    interactive_demo()

