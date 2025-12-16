from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import yaml
import numpy as np

class GlobalGraphRAG:
    def __init__(self, graph_path="data/processed/knowledge_graph.pkl", communities=None, cfg_path="config.yaml"):
        self.G = pickle.load(open(graph_path, "rb"))
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        cfg = yaml.safe_load(open(cfg_path))
        self.top_k = cfg.get('top_k_global', 3)
        self.communities = communities or {}  # cid -> list(nodes)

    def community_summary(self, nodes):
        # simple summary: join representative texts; for richer summary call local LLM summarizer
        texts = [node for node in nodes]
        joined = ". ".join(texts[:20])  # cap length
        # optionally call an LLM here to summarize
        return joined
    
    # def llm_summary()
    def score_point(self, point_text, query):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        p_emb = self.embedder.encode([point_text], convert_to_numpy=True)[0]
        return float(cosine_similarity([q_emb], [p_emb])[0][0])

    def search(self, query, chunks):
        # 1) score communities by similarity between query and community summary
        comm_scores = []
        for cid, nodes in self.communities.items():
            summary = self.community_summary(nodes)
            score = self.score_point(summary, query)
            comm_scores.append((cid, score, summary))
        comm_scores.sort(key=lambda x: x[1], reverse=True)
        # take top-K communities
        top_comms = comm_scores[:self.top_k]
        # 2) from each community, collect chunks (or node text points) and score each point
        scored_points = []
        for cid, cscore, summary in top_comms:
            nodes = self.communities[cid]
            for node in nodes:
                pt = node
                sc = self.score_point(pt, query)
                scored_points.append((pt, sc, cid))
        scored_points.sort(key=lambda x: x[1], reverse=True)
        # map points to chunks if possible (or return node texts)
        return scored_points[:50]
