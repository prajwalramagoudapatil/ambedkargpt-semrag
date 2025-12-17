import networkx as nx
from typing import Dict
from langchain_ollama import OllamaLLM
import json

def get_summary(graph, nodes):
    '''
    implements a simple summary builder by:
        creating a list of entities, and
        "{ subject verb object }" list
    
    modify as per requirments...
    '''

    entity_summary = []
    edge_summary = []
    for n in nodes:
        if 'summary' in graph.nodes[n]:
            entity_summary.append(
                f'entity {n} is {graph.nodes[n]['summary']}.; \n'
            )
    entity_summary = ' '.join(entity_summary)
    for u, v, data in graph.edges(nodes, data = True):
        if 'relations' in data:
            edge_summary.append(
                f' {graph.nodes[u]['summary']}: {u}, relation/verb {graph[u][v]['relations']} {graph.nodes[v]['summary']}: {v};.\n '
            )
    edge_summary = ' '.join(edge_summary)
    return entity_summary, edge_summary


def build_community_summary(graph: nx.Graph,
    communities: Dict , llm_model: str = "gemma3:1b"
) -> str:
    """
    Implements Equation (3) from SemRAG.

    S(C_i) = LLM_summarize( entity summaries, relationship summaries
    )
    """
    llm = OllamaLLM(
            model=llm_model,
            temperature=0.1,
        )
    llm_summary = {}

    for cid, nodes in communities.items():

        nodes_sum, edge_summ = get_summary(graph, nodes)


        prompt = f"""
            You are summarizing a knowledge graph community.

            Below are entity descriptions and relationships.
            Generate a concise, coherent summary capturing the main theme
            and important connections.

            entities data:
            {nodes_sum}
            relations data:
            {edge_summ}
            """
        response = llm.invoke(prompt)

        llm_summary[cid] = response
    
    with open('data/procesed/community_summary.json', mode='w', encoding='utf-8') as file:
        json.dump(obj=llm_summary, fp=file, )
    
