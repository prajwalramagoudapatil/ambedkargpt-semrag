import networkx as nx
import pickle
import json


comm = pickle.load(open('data/processed/communities.pkl', 'rb'))
graph = pickle.load(open('data/processed/knowledge_graph.pkl', 'rb'))
chunks = json.load(open('data/processed/chunks.json', 'r', encoding='utf-8'))



def chunks_for_entity(entity):
    global chunks
    ent_chunks = []
    for idx, ch in enumerate(chunks):
        if entity.lower() in ch.lower():
            ent_chunks.append((idx, ch))

    return ent_chunks


def community_summary():
    community_chunks ={}
    count = 20
    for cid, entities in comm.items():
        print('cid: ', cid, ' +> len: ', len(entities))
        # print('entities: ', entities)
        count -= 1
        if count < 0:
            break

        community_chunks[cid] = {}
        for ent in entities:
            comm_chunks = chunks_for_entity(ent)        # idx, chunk

            for idx, chunk in comm_chunks:
                if idx not in community_chunks[cid]:
                    community_chunks[cid][idx] = chunk
                    
    # print(community_chunks)
    print('len comm: ', len(comm))
    print(len(community_chunks))
    with open('data/processed/summary.json', 'w', encoding='utf-8') as f:
        json.dump(community_chunks, f)
    return community_chunks

community_summary()

