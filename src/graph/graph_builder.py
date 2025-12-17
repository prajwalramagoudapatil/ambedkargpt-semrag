import networkx as nx
import pickle
from .entity_extractor import extract_entities, extract_relations
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

class GraphBuilder:

    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.G = nx.Graph()
        self.embedder = SentenceTransformer(embedding_model)
        self.node_text = {}  # map node -> representative text
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = Path(current_dir)
        if (current_dir / 'data').exists() and (current_dir / 'myenv').exists():
            self.data_dir = str(current_dir / 'data')
        for dir in Path(current_dir).parents:
            if (dir / 'data').exists() and (dir / 'myenv').exists():
                self.data_dir = str(dir / 'data')
                break

    def add_chunk_entities(self, chunk_id, chunk_text):
        ents = extract_entities(chunk_text)
        for ent_text, ent_label in ents:
            if not self.G.has_node(ent_text):
                self.G.add_node(ent_text, label=ent_label)
                self.node_text[ent_text] = ent_text
            # we can also connect chunk as a node, but here we bias entity-entity edges by co-occurrence
        # relations
        rels = extract_relations(chunk_text)
        for s, v, o in rels:
            if not self.G.has_node(s):
                self.G.add_node(s)
                self.node_text[s] = s
                self.G.nodes[s]['summary'] = 'subject'
            if not self.G.has_node(o):
                self.G.add_node(o)
                self.node_text[o] = o
                self.G.nodes[o]['summary'] = 'object'
            # add or increment edge with relation label      summary
            if self.G.has_edge(s, o):
                self.G[s][o].setdefault('weight', 0)
                self.G[s][o]['weight'] += 1
                self.G[s][o].setdefault('relations', []).append(v)
            else:
                self.G.add_edge(s, o, weight=1, relations=[v])

    def finalize(self):
        # compute node embeddings for retrieval later
        print('Creating Embeddings....')
        texts = [self.node_text[n] for n in self.G.nodes()]
        embeds = self.embedder.encode(texts, convert_to_numpy=True)
        for n, emb in zip(self.G.nodes(), embeds):
            self.G.nodes[n]['emb'] = emb

    def save(self, path='data/knowledge_graph.pkl'):
        with open(path, "wb") as f:
            pickle.dump(self.G, f)


            
