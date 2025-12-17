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
                f'entity {n} is {graph.nodes[n]["summary"]}.; \n'
            )
    entity_summary = ' '.join(entity_summary)
    for u, v, data in graph.edges(nodes, data = True):
        if 'relations' in data:
            if 'summary' in graph.nodes[u] and 'summary' in graph.nodes[v]:
                edge_summary.append(
                    f" {graph.nodes[u]['summary']}: {u}, relation/verb {graph[u][v]['relations']} {graph.nodes[v]['summary']}: {v};.\n "
                )
            else:
                edge_summary.append(
                    f"  entity 1: {u}, relation/verb: {graph[u][v]['relations']}  entity 2: {v};.\n "
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
    count = 3
    for cid, nodes in communities.items():
        print('summarizing community ', cid)
        nodes_sum, edge_summ = get_summary(graph, nodes)
        nodes_summ_str = ' '.join(nodes_sum).strip()
        if nodes_summ_str:
            print('Node summ: ', nodes_summ_str)
        else:
            print('node summ: ', nodes_summ_str, ' \n\t*** CONtinueing \t****')
            continue
        edge_summ_str = ' '.join(edge_summ).strip()
        print('Edge summ: ', edge_summ_str[:10])
        prompt = f"""
            You are summarizing a knowledge graph community.

            Below are entity descriptions and relationships.
            Generate a concise, coherent summary capturing the main theme
            and important connections.

            entities data:
            {nodes_summ_str}
            relations data:
            {edge_summ_str}
            """
        # print('\t prompt: ', prompt, '\n ')
        response = llm.invoke(prompt)
        print(cid, ' summary: ', response)
        llm_summary[cid] = response
        count -= 1
        if count < 0:
            break
    
    with open('data/processed/community_summary.json', mode='w', encoding='utf-8') as file:
        json.dump(obj=llm_summary, fp=file, )
    
    print('Saved summary.!! :-)')
