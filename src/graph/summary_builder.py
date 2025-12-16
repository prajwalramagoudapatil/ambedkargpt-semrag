import networkx as nx
from typing import List
from langchain_ollama import ChatOllama

'summary'

def get_summary(graph, nodes):

    entity_summary = []
    edge_summary = []
    for n in nodes:
        if 'summary' in graph.nodes[n]:
            entity_summary.append(
                f'entity {n} is {graph.nodes[n]['summary']}'
            )
    entity_summary = ' '.join(entity_summary)
    for u, v, data in graph.edges(nodes, data = True):
        if 'relations' in data:
            edge_summary.append(
                f' {graph.nodes[u]['summary']}: {u}, relation/verb {graph[u][v]['relations']} {graph.nodes[v]['summary']}: {v} '
            )
    edge_summary = ' '.join(edge_summary)
    return entity_summary, edge_summary


def build_community_summary(graph: nx.Graph,
    communities , llm_model: str = "llama3:8b"
) -> str:
    """
    Implements Equation (3) from SemRAG.

    S(C_i) = LLM_summarize( entity summaries, relationship summaries
    )
    """
    
    llm_summary = {}

    for cid, nodes in communities.items():

        nodes_sum, edge_summ = get_summary(nodes)


        prompt = f"""
            You are summarizing a knowledge graph community.

            Below are entity descriptions and relationships.
            Generate a concise, coherent summary capturing the main theme
            and important connections.

            nodes data:
            {nodes_sum}
            edges data:
            {edge_summ}
            """
        llm = ChatOllama(
            
        )

