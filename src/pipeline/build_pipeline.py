import json, pickle
import sys
from pathlib import Path

current_file_dir = Path(__file__).resolve().parent # directory of ambedkargpt.py = src/pipeline
src_directory = current_file_dir.parent # src directory
data_dir = src_directory.parent / 'data'

ambedkar_dir = src_directory.parent

if Path(ambedkar_dir).exists:
    sys.path.insert(0, str(ambedkar_dir))

from src.chunking.semantic_chunker import SemanticChunker
from src.graph.graph_builder import GraphBuilder
from src.graph.community_detector import detect_communities
from src.graph.summary_builder import build_community_summary

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
    # build_pipeline()
    chunks = json.load(open(data_dir/'processed/chunks.json', 'r', encoding='utf-8'))
    print("Building knowledge graph...")
    gb = GraphBuilder()
    for i, ch in enumerate(chunks):
        gb.add_chunk_entities(i, ch)
    gb.finalize()
    print("Saving knowledge graph...")
    gb.save( data_dir / "processed/knowledge_graph.pkl")
    # # 3. communities
    # G = gb.G
    # print("Detecting communities...")
    # comms = detect_communities(G)
    # print("Saving communities...")
    # pickle.dump(comms, open("data/processed/communities.pkl","wb"))
    # print('Create community summary')
    # build_community_summary(graph=G, communities=comms, llm_model='gemma3:1b')
