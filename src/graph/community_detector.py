import networkx as nx
import community as community_louvain  # python-louvain; alternative: leidenalg (igraph) if available

def detect_communities(G):
    # partition is dict node->community_id
    partition = community_louvain.best_partition(nx.convert_matrix.to_numpy_matrix(G) if False else G)
    # group nodes by community
    communities = {}
    for n, cid in partition.items():
        communities.setdefault(cid, []).append(n)
    return communities