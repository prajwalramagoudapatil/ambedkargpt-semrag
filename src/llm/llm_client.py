from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq

from typing import List
import json

with open("config.json", "r", encoding='utf-8') as f:
    config = json.load(f)

api_key = config.get('groq_api_key', '')

def build_prompt(query: str, local_results: List[dict], global_points: List[tuple]):
    ctx = []
    ctx.append("You are an assistant answering questions about Dr. B.R. Ambedkar's works. Use the context and cite chunk indexes when possible.\n")
    ctx.append("Query: " + query + "\n")
    ctx.append("Local results:\n")
    for r in local_results:
        ctx.append(f"[CHUNK {r['chunk_idx']}] {r['chunk_text'][:500]}...\n")
    ctx.append("\nGlobal points:\n")
    for pt, score, cid in global_points[:10]:
        ctx.append(f"[NODE {cid}] {pt}\n")
    ctx.append("\nAnswer concisely and include citations like [CHUNK 12]. If information is insufficient, say 'Insufficient information'.")
    return "\n".join(ctx)

def generate_with_langchain_ollama(prompt: str, model_name: str = "llama3.2:1b"):
    ollama = OllamaLLM(
        model=model_name,
        temperature=0.1
        )
    response = ollama.invoke(prompt)
    return response

def generate_with_langchain_groq(prompt: str, model_name: str = "llama-3.1-8b-instant"):
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0.2
    )
    response = llm.invoke(prompt)
    return response.content

def generate_answer(query, local_results, global_points):
    prompt = build_prompt(query, local_results, global_points)
    # call local model:

    answer = generate_with_langchain_ollama(prompt)
    return answer
