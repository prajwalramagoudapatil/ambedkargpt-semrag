from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from typing import List
import json

with open("config.json", "r", encoding='utf-8') as f:
    config = json.load(f)

api_key = config.get('groq_api_key', '')
model_name: str = "llama3.2:1b"

ollama = ChatOllama(
    model=model_name,
    temperature=0.1
)

def build_prompt(query: str, local_results: List[dict], global_points: List[tuple]):

    local_res = []
    global_res = []
    for r in local_results:
        local_res.append(f"[CHUNK {r['chunk_idx']}] {r['chunk_text'][:500]}...\n")

    for pt, score, cid in global_points[:20]:
        global_res.append(f"[NODE {cid}] {pt}\n")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant answering questions about Dr. B.R. Ambedkar's works. Use the context and cite chunk indexes when possible. "
         "Local results: {local_results} "
         "Global points: {global_points} "
        
         "Answer concisely and include citations like [CHUNK 12]. If information is insufficient, say 'Insufficient information'. "
        ),
        ("human", "{question}")
    ])

    return prompt.invoke({'local_results':local_res, 'global_points':global_res, 'question': query})


def generate_with_langchain_ollama(prompt, model_name: str = "llama3.2:1b"):

    response = ollama.invoke(prompt)
    return response.content

# def generate_with_langchain_groq(prompt: str, model_name: str = "llama-3.1-8b-instant"):
#     llm = ChatGroq(
#         groq_api_key=api_key,
#         model=model_name,
#         temperature=0.2
#     )
#     response = llm.invoke(prompt)
#     return response.content

def generate_answer(query, local_results, global_points):
    prompt = build_prompt(query, local_results, global_points)
    # call local model:

    answer = generate_with_langchain_ollama(prompt)
    return answer
