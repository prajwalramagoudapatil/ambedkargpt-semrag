import json, pickle
from chunking.semantic_chunker import SemanticChunker
from graph.graph_builder import GraphBuilder
from graph.community_detector import detect_communities
from llm.llm_client import generate_answer
from pathlib import Path
from graph.summary_builder import build_community_summary

current_file_dir = Path(__file__).resolve().parent # directory of ambedkargpt.py = src/pipeline

src_directory = current_file_dir.parent # src directory

src_path_str = str(src_directory)
data_dir = src_directory.parent / 'data'

def build_pipeline(pdf_path= (data_dir / "Ambedkar_book.pdf")):
    # 1. Chunk
    print("Starting chunking process...")
    chunker = SemanticChunker("config.yaml")
    chunks = chunker.process_pdf_to_chunks(pdf_path, out_path= (data_dir / "processed/chunks.json"))
    # 2. Graph
    print("Building knowledge graph...")
    gb = GraphBuilder()
    for i, ch in enumerate(chunks):
        gb.add_chunk_entities(i, ch)
    gb.finalize()
    print("Saving knowledge graph...")
    gb.save( data_dir / "processed/knowledge_graph.pkl")
    # 3. communities
    G = gb.G
    print("Detecting communities...")
    comms = detect_communities(G)
    print("Saving communities...")
    pickle.dump(comms, open("data/processed/communities.pkl","wb"))
    print('Create community summary')
    build_community_summary(graph=G, communities=comms, llm_model='gemma3:1b')
    
    return chunks, G, comms

if __name__ == '__main__':
    build_pipeline()