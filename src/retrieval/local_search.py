import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

class LocalGraphRAG:
    def __init__(self, graph_path="data/processed/knowledge_graph.pkl", cfg_path="config.yaml"):
        self.G = pickle.load(open(graph_path, "rb"))
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        cfg = __import__("yaml").safe_load(open(cfg_path))
        self.tau_e = cfg.get('tau_e', 0.6)
        self.tau_d = cfg.get('tau_d', 0.55)
        self.top_k = cfg.get('top_k_local', 5)

    def _embed_query(self, q):
        return self.embedder.encode([q], convert_to_numpy=True)[0]

    def entity_similarity(self, q_emb):
        sims = []
        for n in self.G.nodes():
            emb = self.G.nodes[n].get('emb')
            if emb is None:
                continue
            s = float(cosine_similarity([q_emb], [emb])[0][0])
            sims.append((n, s))     # (node, similart )
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims         # return List(node, simil with query)

    def find_chunks_for_entity(self, entity, chunks):
        # naive: return chunks containing the entity string
        res = []
        for i, ch in enumerate(chunks):
            if entity.lower() in ch.lower():
                res.append((i, ch))
        return res

    def search(self, query, chunks):
        q_emb = self._embed_query(query)
        ent_sims = self.entity_similarity(q_emb)
        # filter by tau_e
        ent_sims = [ (n, s) for n,s in ent_sims if s >= self.tau_e ]
        chunk_scores = {}  # chunk_idx -> best score
        chunk_data = {}    # chunk_idx -> stored metadata
        for ent, ent_score in ent_sims[:self.top_k]:
            related_chunks = self.find_chunks_for_entity(ent, chunks)
            # for each chunk compute sim(g,v) (approx: chunk embedding vs entity embedding)
            for idx, ch in related_chunks:
                ch_emb = self.embedder.encode([ch], convert_to_numpy=True)[0]
                v_emb = self.G.nodes[ent]['emb']
                sim_gv = float(cosine_similarity([ch_emb], [v_emb])[0][0])
                
                if sim_gv < self.tau_d:
                    continue

                # combined score (Equation 4 style)
                final_score = 0.6 * ent_score + 0.4 * sim_gv

                # Deduplicate by chunk_idx
                if idx not in chunk_scores or final_score > chunk_scores[idx]:
                    chunk_scores[idx] = final_score
                    chunk_data[idx] = {
                        "chunk_idx": idx,
                        "chunk_text": ch,
                        "score": final_score,
                        "entities": [ent]
                    }
                else:
                    # same chunk from different entity
                    chunk_data[idx]["entities"].append(ent)

        # Sort by score
        ranked = sorted(
            chunk_data.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:self.top_k]